"""REQ-REPORT-8239: authenticate a current seal before exposing evaluator targets.

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
from carnot.verify import margin_decision_audit_8239 as numeric
from carnot.verify import utility_fit_execution_8222 as upstream
from carnot.verify import utility_patch_methods_8219 as frozen
from carnot.reporting import decision_margin_methods_8234 as methods

Json = dict[str, Any]
ROOT = frozen.ROOT
NAME = "experiment_8239_v712_margin_decision_audit"
TASK = "exp8239-margin-decision-audit"
MODULE = "python/carnot/verify/margin_decision_audit_execution_8239.py"
RUNNER = MODULE
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_margin_decision_audit_8239.py"
OWNED = ["python/carnot/verify/margin_decision_audit_8239.py", MODULE, CLI]
RUN_DATE = "20261007"
MODEL_SPECS: list[Json] = []
PROTOCOL, PIN = methods.PROTOCOL, methods.PIN
UPSTREAM = "results/experiment_8238_v712_margin_prediction_seal.json"
METHODS = "results/experiment_8234_v712_decision_margin_methods.json"
LABELS = "results/raw/experiment_8185_v707_sentence_decision_audit/invocations/1791261147226658146/measurement.json"
PINS = {
    METHODS: "sha256:018d57444d13ec1402de739b5141bc7df57a6edc4004e097c9caa31968418f37",
    UPSTREAM: "sha256:0da40b8bb9b61f0b93428f5ce69a72b77cfa891461406db1f0e95c1b65c138e3",
    LABELS: "sha256:08d8a8efe35c0c44e49ee14e761264f20b70b1b665e1d706b20bb94047d81e44",
}
reference = frozen.reference


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries and real counts so cached reduction never looks stalled."""
    print(f"[exp8239] phase={phase} completed={completed} pending={pending}", flush=True)


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

    read(root / PROTOCOL, PIN)
    method = read(root / METHODS, PINS[METHODS])
    for field, expected in [
        ("schema", "carnot.v712.decision-margin-methods.v1"),
        ("margin_protocol_ready_score", 1),
        ("protocol_sha256", PIN),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
    ]:
        frozen.gate(work, root / METHODS, field, expected, method.get(field))
    value = read(root / UPSTREAM, PINS[UPSTREAM])
    for field, expected in [
        ("schema", "carnot.v712.margin-prediction-seal.v1"),
        ("protocol_sha256", PIN),
        ("margin_predictions_ready_score", 1),
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
        frozen.bind(work, Path(ref["path"]), ref["sha256"], raw)
    for ref in value["code_config_hashes"]:
        frozen.bind(work, Path(ref["snapshot_path"]), ref["sha256"], raw)
    saved = read(
        Path(value["measurement_reference"]["path"]), value["measurement_reference"]["sha256"]
    )
    predictions = read(Path(value["predictions_path"]), value["predictions_sha256"])
    frozen.gate(
        work,
        Path(value["predictions_path"]),
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
        with TemporaryDirectory(prefix="carnot-8239-probe-") as directory:
            probe = Path(directory) / "probe"
            probe.write_bytes(b"private writable scratch")
            frozen.gate(
                work,
                probe,
                "private_scratch_writable",
                True,
                probe.read_bytes() == b"private writable scratch"
                and probe.parent.stat().st_mode & 0o777 == 0o700,
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
            PROTOCOL,
            "python/carnot/reporting/v709_execution.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/verify/margin_prediction_seal_8238.py",
            "python/carnot/verify/margin_energy_training_8237.py",
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
    """A reproducible null qualifies execution while retiring only this objective.

    The source cohort was used in development. Passing evaluation cannot turn
    these sources into independent generalization evidence.
    """
    from scripts.experiment_template import normalize_artifact_for_template_write

    failures = [c for c in work["checks"] if not c["passed"]]
    checked = bool(receipts) and all(r["passed"] for r in receipts) and not work["owned_failure"]
    result = work["diagnostics"] or numeric.statistics([], dict(arm="energy_global"))
    ready = int(checked and not failures and bool(work["diagnostics"]))
    signal = ready * result["h1_development_signal_score"]
    verdict = (
        "disqualified"
        if not checked
        else "blocked"
        if failures
        else "positive"
        if signal
        else "null"
    )
    operand = (
        Path(failures[0]["path"]).stem.lower() + "_" + failures[0]["artifact_field"]
        if failures
        else "margin_decision_audit"
    )
    rows = result.get(
        "rows",
        [
            dict(
                r,
                slot=i + 1,
                arm=a,
                seed=None,
                condition="original_reserved_slot",
                status="excluded",
                metric="typed_decision_cost",
                numerator=None,
                denominator=0,
                missing_evidence=True,
                exclusion_reason="external_operand_unavailable"
                if failures
                else "owned_audit_failure",
            )
            for i, r in enumerate(numeric.PROTOCOL["role_manifest"]["reserved"])
            for a, _ in numeric.ARM_KEYS
        ],
    )
    value: Json = dict(
        result,
        experiment_id=8239,
        task_id=TASK,
        milestone="2026.10.712",
        run_date=RUN_DATE,
        schema="carnot.v712.margin-decision-audit.v1",
        honest_verdict="complete_" + verdict + "_" + operand,
        verdict_class=verdict,
        gate_check_summary=work["checks"],
        inference_substrate="aggregation_from_upstream_artifacts",
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
        exposure_scope="exposed_development_within_run_prediction_label_boundary",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        margin_audit_ready_score=ready,
        h1_development_signal_score=signal,
        energy_specific_advantage_score=ready * result["energy_specific_advantage_score"],
        trained_head_specs=[
            dict(arm=a, coefficients=17, current_fit=False, generator=False)
            for a in numeric.seal.ARMS
        ],
        owned_failure=work["owned_failure"],
        required_checks_passed=checked,
        flagged_adversarial=False,
        acceptance_gates=dict(
            external_inputs=not failures,
            owned_validation=checked,
            scientific_H1=bool(result["H1"]["passed"]),
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=[
            dict(resource=c["check"], available=c["passed"], path=c["path"]) for c in work["checks"]
        ],
        precondition_receipts=work["precondition_receipts"],
        duration_s=work["duration_s"],
        random_seed=numeric.H1["seed"],
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"],
        phase_spans=[
            dict(phase="authenticate_and_reduce", **work["clock"], duration_s=work["duration_s"])
        ],
        measurement_clocks=work["clock"],
        measurement_reference=reference(raw / "measurement.json"),
        protocol_path=str(ROOT / PROTOCOL),
        protocol_sha256=PIN,
        frozen_comparator=work["evidence"].get("comparator"),
        fixture_protocol_only=fixture,
        repository_health=work.get("global_health", {}),
        changed_code_coverage_reference=[
            reference(p)
            for p in [
                raw.parents[1] / "owned_coverage.json",
                raw.parents[1] / "frozen_coverage_config.ini",
            ]
            if p.is_file()
        ],
        mechanism_disposition=dict(
            status="retired_exact_margin_only_objective"
            if ready and not signal
            else "development_signal_requires_independent_confirmation"
            if signal
            else "evaluation_unqualified",
            scope="exposed_development_cohort",
            objective="native-margin-weighted binary log loss with unchanged extraction",
            recommendation="Collect new independent source evidence or change the extraction mechanism before another calibration variant.",
            broad_impossibility_claim=False,
            independent_generalization_success=False,
        ),
        claim_scope="Margin-trained decision value on exposed development sources only.",
        methodology_note="No model loads or current LLM calls. Authenticated frozen six-head predictions and original evaluator annotations determine all-slot typed costs, complete-case Brier and false accepts. Missing masks retain128 original sources. Paired source bootstrap uses10000 draws, seed7128239 and one-sided97.5 percent confidence. Matched simple-head controls separate shared weighting from energy-specific advantage. Historical Qwen provenance is not a current call. Both generalization scores remain zero.",
    )
    imported = {
        str(ROOT / UPSTREAM): [
            "frozen_comparator",
            "predictions_path",
            "predictions_sha256",
            "measurement_reference",
            "margin_predictions_ready_score",
            "required_checks_passed",
            "flagged_adversarial",
            "labels_opened",
            "terminal_validation_sidecar_path",
        ],
        str(ROOT / LABELS): ["evidence.capture.slots", "evidence.original_response_records"],
    }
    value["cited_upstream_artifacts"] = [
        dict(
            path=r["upstream_path"],
            sha256=r["sha256"],
            fields_imported=imported.get(r["upstream_path"], ["authenticated_byte_snapshot"]),
        )
        for r in work["refs"]
    ]
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Bind actual invocation, authenticated primitive evidence, original sources and measured checks; readiness grants no scientific credit."
        for k in value
    }
    value["field_principles"].update(
        margin_audit_ready_score="One requires authenticated operands and all owned checks; a measured null can be ready.",
        H1="Every frozen primary benefit and harm gate must pass on exposed development sources.",
        energy_specific_advantage_score="Readiness and H1 plus lower cost-gain bounds above0.02 against both equally weighted simple heads.",
        per_source_deltas="All128 source slots retain missing status, paired costs and every arm delta.",
        action_switch_rows="Probability movement is descriptive; actual switches and cost changes determine benefit.",
        bootstrap_diagnostics="Paired source draws preserve128 slots and fixed missing masks; seeds add no sources.",
        mechanism_disposition="A qualified null retires only this margin-only objective on the exposed cohort.",
    )
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
            predictions = load(source["predictions_path"], source["predictions_sha256"])
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
            spec["argv"].insert(-1, "tests/python/test_margin_prediction_seal_8238.py")
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
