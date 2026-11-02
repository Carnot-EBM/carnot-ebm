"""REQ-REPORT-8210: authenticate sealed actions before independent human costs.

This invocation opens only cached evidence. Historic labels have been reused,
so even a measured gain cannot establish independent generalization.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import restricted_decision_rule_8210 as n
from carnot.verify import restricted_sealed_evaluation_8209 as sealed
from carnot.verify import selective_decision_audit_8197 as historical
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT, fit, reference, execution = sealed.ROOT, sealed.fit, sealed.reference, sealed.execution
NAME = "experiment_8210_v709_restricted_decision_audit"
TASK = "exp8210-restricted-decision-audit"
MODULE = "python/carnot/verify/restricted_decision_audit_8210.py"
NUMERIC = "python/carnot/verify/restricted_decision_rule_8210.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_restricted_decision_audit_8210.py"
OWNED = [MODULE, NUMERIC, CLI]
RUN_DATE = "20261006"
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8209_v709_restricted_sealed_evaluation.json"
LABELS = historical.HISTORICAL
OPTIONAL = "results/experiment_8197_v708_selective_decision_audit.json"
PINS = {
    UPSTREAM: "sha256:6b72ec32d865f490f3309b673c671ada3212b99b18e3b6352d147cfce843b579",
    LABELS: historical.PINS[LABELS],
}
OPTIONAL_PIN = "sha256:5a23816e348c5f0f0e92002e01cc699c0beb356a9aecb03f1be1d84c6ac11bf0"
CONFIG = dict(H1=n.H1, costs=n.COSTS)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts distinguish real reductions from a waiting validation child."""
    print(f"[exp8210] phase={phase} completed={completed} pending={pending}", flush=True)


def fixture() -> Json:
    """Original annotations qualify joins without adding natural scientific credit."""
    source, public = historical.fixture(), sealed.fixture()
    slots = source["slots"]
    for feature, slot in zip(public["features"], slots, strict=True):
        feature.update(
            source_sha256=canonical_hash(slot["source_bytes"]),
            answer_sha256=canonical_hash(slot["answer_bytes"]),
        )
    return dict(
        sealed=public,
        slots=slots,
        original_response_records=source["original_response_records"],
        predictions=sealed.n.reduce(public)["prediction_rows"],
        clock=dict(predictions_sealed_ns=1, labels_opened_ns=2),
        **deepcopy(CONFIG),
    )


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """Required operands stop on failure; optional prior nulls remain dispositions."""
    began, wall = time.monotonic_ns(), time.time_ns()
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(
        checks=[],
        refs=[],
        raw_shard_hashes=[],
        evidence={},
        owned_failure="",
        precondition_receipts=[],
        historical_null={},
        optional_sibling_disposition="not_requested_in_fixture",
    )
    gate = lambda path, field, expected, observed: fit.gate(work, path, field, expected, observed)
    read = lambda ref: fit.bind(work, ref, raw)
    progress("before_preconditions")
    try:
        probe = raw / ".probe"
        probe.write_bytes(b"private scratch")
        gate(raw, "private_scratch_writable", True, probe.read_bytes() == b"private scratch")
        probe.unlink()
        if mutation:
            gate(root / UPSTREAM, "sealed_action_ready_score", 1, None)
        if fixture:
            data = globals()["fixture"]()
        else:
            command = sealed.upstream.methods.supervisor.child(
                "python_environment",
                [
                    sys.executable,
                    "-c",
                    "import json,os,sys,pytest,coverage,scipy,numpy; from pathlib import Path; "
                    "root=Path(sys.argv[1]); tools={n:os.access(root/'.venv/bin'/n,os.X_OK) for n in ['python','pytest','coverage','ruff','mypy']}; "
                    "print(json.dumps({'python':sys.version,'executable':sys.executable,'tools':tools,'pytest':pytest.__version__,'coverage':coverage.__version__,'scipy':scipy.__version__,'numpy':numpy.__version__,'named_paths':{p:(root/p).is_file() for p in sys.argv[2:]}})); "
                    "raise SystemExit(int(not all(tools.values())))",
                    str(root),
                    UPSTREAM,
                    sealed.UPSTREAM,
                    sealed.upstream.UPSTREAM,
                    LABELS,
                    OPTIONAL,
                    sealed.upstream.methods.PROTOCOL,
                ],
                raw / "preconditions",
                deadline=30,
            )
            work["precondition_receipts"].append(command)
            gate(Path(sys.executable), "python_environment_exit", 0, command["exit_code"])
            value = read(dict(path=str(root / UPSTREAM), sha256=PINS[UPSTREAM]))
            for key, expected in [
                ("sealed_action_ready_score", 1),
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
                ("evaluator_targets_opened", False),
            ]:
                gate(root / UPSTREAM, key, expected, value.get(key))
            gate(
                root / UPSTREAM,
                "terminal_publication_passed",
                True,
                read_bound_sidecar(root / UPSTREAM, fit.historical.publication_sidecar(value))[
                    "report"
                ]["passed"],
            )
            gate(root / UPSTREAM, "sealed_cold_replay", True, sealed.replay(root / UPSTREAM))
            saved = read(value["measurement_reference"])
            predictions = read(
                dict(path=value["predictions_path"], sha256=value["predictions_sha256"])
            )
            gate(
                Path(value["predictions_path"]),
                "labels_opened",
                False,
                predictions["labels_opened"],
            )
            ledger = value["label_access_ledger"]
            gate(
                root / UPSTREAM,
                "label_access_order",
                True,
                [r["event"] for r in ledger]
                == ["roster_frozen", "target_free_features_opened", "predictions_sealed"]
                and all(not r["labels_opened"] for r in ledger)
                and all(a["monotonic_ns"] < b["monotonic_ns"] for a, b in zip(ledger, ledger[1:])),
            )
            protocol = read(
                dict(
                    path=str(ROOT / sealed.upstream.methods.PROTOCOL),
                    sha256=sealed.upstream.methods.PIN,
                )
            )
            gate(
                ROOT / sealed.upstream.methods.PROTOCOL,
                "registered_H1",
                CONFIG,
                dict(H1=protocol["H1"], costs=protocol["costs"]),
            )
            data = dict(
                sealed=saved["evidence"], predictions=predictions["rows"], **deepcopy(CONFIG)
            )
            historical.old.base.audit.equal(
                sealed.n.reduce(data["sealed"])["prediction_rows"], data["predictions"]
            )
            atomic_json(raw / "authenticated_predictions.json", predictions)
            clock = dict(predictions_sealed_ns=time.monotonic_ns())
            progress("predictions_authenticated_before_label_access", 768, 0)
            clock["labels_opened_ns"] = time.monotonic_ns()
            prior = read(dict(path=str(root / LABELS), sha256=PINS[LABELS]))
            labels = read(prior["measurement_reference"])["evidence"]
            data.update(
                slots=labels["capture"]["slots"],
                original_response_records=labels["original_response_records"],
                clock=clock,
            )
            optional = root / OPTIONAL
            if optional.is_file() and sha256_file(optional) == OPTIONAL_PIN:
                old = read(dict(path=str(optional), sha256=OPTIONAL_PIN))
                work["historical_null"] = {
                    k: old[k]
                    for k in (
                        "verdict_class",
                        "honest_verdict",
                        "H1",
                        "selective_audit_ready_score",
                    )
                }
                work["optional_sibling_disposition"] = "authenticated_historical_null_context_only"
            else:
                work["optional_sibling_disposition"] = (
                    "optional_historical_context_absent_or_hash_changed"
                )
        progress("after_preconditions", 128, 0)
        work["owned_failure"] = "reduction_incomplete"
        progress("before_benchmark_independent_reduction", 0, 128)
        reduced = n.reduce(data)
        progress("after_benchmark_independent_reduction", 128, 0)
        for name, content in [
            ("primitive_evidence", data),
            ("independent_reduction", reduced),
            ("frozen_configuration", CONFIG),
        ]:
            path = raw / (name + ".json")
            atomic_json(path, content)
            work["raw_shard_hashes"].append(reference(path))
        work.update(evidence=data, owned_failure="")
    except (OSError, ValueError, KeyError, TypeError) as error:
        if work["owned_failure"]:
            work.update(owned_failure=str(error), evidence={})
        elif all(c["passed"] for c in work["checks"]):
            work["checks"].append(
                dict(
                    check="input_structure",
                    upstream=UPSTREAM,
                    path=str(root / UPSTREAM),
                    hash=None,
                    artifact_field="input_structure",
                    op="==",
                    expected="authenticated_original_evidence",
                    observed=str(error),
                    passed=False,
                )
            )
        progress("terminal_operand_or_owned_failure", len(work["checks"]), 0)
    work.update(
        duration_s=(time.monotonic_ns() - began) / 1e9,
        measurement_clocks=dict(
            started_monotonic_ns=began, ended_monotonic_ns=time.monotonic_ns(), started_wall_ns=wall
        ),
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                sealed.MODULE,
                sealed.NUMERIC,
                historical.MODULE,
                sealed.upstream.methods.PROTOCOL,
                "scripts/experiment_template.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/v709_execution.py",
                "python/carnot/reporting/methods_stream_execution_8111.py",
                "python/carnot/verify/sentence_decision_audit_8185.py",
            ]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """A valid null is ready; a failed owned check cannot retain audit readiness."""
    checked = bool(receipts) and all(r["passed"] for r in receipts) and not work["owned_failure"]
    failures = [c for c in work["checks"] if not c["passed"]]
    reduced = (
        n.reduce(work["evidence"])
        if work["evidence"]
        else dict(
            rows=[],
            intended_count=128,
            completed_count=0,
            independent_count=0,
            failed_count=0,
            censored_count=0,
            excluded_count=128,
            equivalent_logistic_parity=dict(passed=False),
            **n.statistics([], "logistic"),
        )
    )
    ready = int(checked and not failures and bool(work["evidence"]))
    signal = ready * reduced["h1_development_signal_score"]
    verdict = (
        "disqualified"
        if not checked
        else "blocked"
        if failures
        else "circular_positive"
        if fixture and signal
        else "positive"
        if signal
        else "null"
    )
    value: Json = dict(
        experiment_id=8210,
        task_id=TASK,
        milestone="2026.10.709",
        run_date=RUN_DATE,
        honest_verdict="complete_" + verdict + "_restricted_decision_audit",
        verdict_class=verdict,
        gate_check_summary=work["checks"],
        preconditions_checked=[
            dict(resource=c["check"], available=c["passed"], path=c["path"]) for c in work["checks"]
        ],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(
            model_loads=0, generate_calls=0, forward_calls=0, live_model_calls=0, model_count=0
        ),
        call_ledger=[],
        duration_s=work["duration_s"],
        random_seed=n.H1["seed"],
        source_artifact_hashes=work["refs"],
        cited_upstream_artifacts=[
            dict(
                path=p,
                sha256=s,
                fields_imported=[
                    "sealed_prediction_custody"
                    if p == UPSTREAM
                    else "original_source_annotation_bytes"
                ],
            )
            for p, s in PINS.items()
        ]
        if not fixture
        else [],
        verifier_is_oracle=fixture,
        exposure_scope="oracle_fixture_only"
        if fixture
        else "exposed_development_within_run_prediction_label_boundary",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        acceptance_gates=dict(
            external_inputs=not failures,
            owned_validation=bool(checked),
            scientific_H1=reduced["H1"]["passed"],
        ),
        validation_receipts=receipts,
        precondition_receipts=work["precondition_receipts"],
        required_checks_passed=bool(checked),
        flagged_adversarial=False,
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        owned_coverage_evidence=[
            reference(p)
            for p in [
                raw.parents[1] / "owned_coverage.json",
                raw.parents[1] / "frozen_coverage_config.ini",
            ]
            if p.is_file()
        ],
        action_audit_ready_score=ready,
        measurement_reference=reference(raw / "measurement.json"),
        measurement_clocks=work["measurement_clocks"],
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        repository_health=work.get("global_health", {}),
        optional_sibling_disposition=work["optional_sibling_disposition"],
        historical_v708_null=work["historical_null"],
        local_set_retirement="unchanged_class_conditional_singleton_policy_remains_retired; historical_primary_bytes_immutable",
        mutation_rejection_evidence=dict(
            scope="private_original_evidence_and_real_cold_CLI_tests",
            required_check="owned_unit_and_private_CLI",
            mutations=["row", "original_label", "permission_mask", "denominator"],
        ),
        sample_size_budget=CONFIG,
        methodology_note="Independent original human targets and frozen typed costs; all128 slots include escalation. Only registered H1 is tested against the frozen tune-selected equally restricted simple control. Policy versus original is descriptive. Equivalent logistic is an identity arm. Reused development data cannot support population safety or independent generalization.",
        claim_scope="Exposed development source utility only; qualified negative evidence remains available.",
        **reduced,
    )
    value["h1_development_signal_score"] = signal
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Bind original sources, frozen operands, real exits and exact invocation; audit readiness is separate from H1 and generalization."
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Pinned source copies defeat primitive edits even after outer rehashing."""
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
            *value["owned_coverage_evidence"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for receipt in [*value["validation_receipts"], *value["precondition_receipts"]]:
            for label in ("stdout", "stderr"):
                if (
                    label + "_path" in receipt
                    and sha256_file(Path(receipt[label + "_path"])) != receipt[label + "_sha256"]
                ):
                    return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_bytes())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        data = work["evidence"]
        if data:
            if not value["verifier_is_oracle"]:
                copies = {Path(r["path"]).name: r for r in work["refs"]}

                def load(ref: Json) -> Json:
                    """Original expected hashes select the copied bytes rather than edited refs."""
                    named = ref["sha256"][7:] + "-" + Path(ref["path"]).name
                    copy = copies[named]
                    if copy["sha256"] != ref["sha256"]:
                        raise ValueError("pinned_input_changed")
                    return dict(json.loads(Path(copy["path"]).read_bytes()))

                source = load(dict(path=UPSTREAM, sha256=PINS[UPSTREAM]))
                prior = load(dict(path=LABELS, sha256=PINS[LABELS]))
                labels = load(prior["measurement_reference"])["evidence"]
                if (
                    data["sealed"] != load(source["measurement_reference"])["evidence"]
                    or data["predictions"]
                    != load(
                        dict(path=source["predictions_path"], sha256=source["predictions_sha256"])
                    )["rows"]
                    or data["slots"] != labels["capture"]["slots"]
                    or data["original_response_records"] != labels["original_response_records"]
                ):
                    return False
            for name, expected in [
                ("primitive_evidence", data),
                ("independent_reduction", n.reduce(data)),
                ("frozen_configuration", CONFIG),
            ]:
                if json.loads((raw / (name + ".json")).read_bytes()) != expected:
                    return False
        return build(
            work, raw, value["validation_receipts"], fixture=value["verifier_is_oracle"]
        ) == dict(value, reproducibility_checksum=checksum)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze exact owned commands and retain a single full-suite health receipt."""
    with (
        patch.object(sealed.upstream, "OWNED", OWNED),
        patch.object(sealed.upstream, "TEST", TEST),
        patch.object(sealed.upstream, "CLI", CLI),
        patch.object(execution, "e", sys.modules[__name__]),
    ):
        specs = sealed.upstream.manifest(private, candidate)
    (candidate.parent / "frozen_coverage_config.ini").write_bytes(
        (private / "coverage.ini").read_bytes()
    )
    for spec in specs["commands"]:
        if spec["name"] == "consumer_and_E2E015_019":
            spec["argv"].insert(-1, "tests/python/test_restricted_sealed_evaluation_8209.py")
        if spec["name"] == "coverage_json":
            spec["argv"][-1] = str(candidate.parent / "owned_coverage.json")
    return specs


def main(argv: list[str] | None = None) -> int:
    """The qualified runner validates private candidate bytes before atomic publication."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
        patch.object(execution, "manifest", manifest),
        patch.object(execution, "run_check", sealed.upstream.methods.run_check),
    ):
        return int(execution.main(argv))
