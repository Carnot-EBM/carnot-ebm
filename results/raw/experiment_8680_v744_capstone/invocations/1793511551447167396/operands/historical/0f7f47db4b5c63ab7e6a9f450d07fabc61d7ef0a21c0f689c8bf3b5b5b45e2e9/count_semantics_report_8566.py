"""REQ-REPORT-8566: publish a bug study without granting scientific benefit.

Readiness describes complete regression controls. It cannot qualify Exp8555 or
turn reused observations into independent learner evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from carnot.reporting import count_semantics_8566 as e
from carnot.reporting.count_semantics_evidence_8566 import read, reference
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash

Json = dict[str, Any]
REQUIRED = {
    "original_adversarial",
    "owned",
    "e2e018",
    "e2e021",
    "ruff_check",
    "ruff_format",
    "mypy",
    "spec",
    "coverage",
    "red_harness",
    "collection",
    "adversarial",
    "row_lint",
    "cold",
    "missing",
    "owned_failure",
    "rehashed",
}


def checksum(value: Json) -> str:
    """Exclude only the checksum itself so every published field stays bound."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def build(work: Json, output: Path) -> Json:
    """Grant readiness only when complete primitives and every required child pass."""
    evidence = work["evidence"]
    classifications = e.classify(work["corpus"])
    receipts = work["receipts"]
    gates = dict(
        preconditions=work["preconditions"]["writable"]
        and all(work["preconditions"]["tools"].values())
        and work["preconditions"]["free_disk_bytes"] > 100000000,
        original_evidence=not evidence["failures"],
        frozen_corpus=work["corpus"] == e.corpus(),
        complete_expectations=all(r["expectation_met"] for r in classifications),
        seven_proofs=len(evidence["cardinality_reductions"]) == 7
        and all(r["proven"] for r in evidence["cardinality_reductions"]),
        historical_controls=len(evidence["historical_controls"]) == 2,
        exact_node_union=work.get("union_passed", False),
        required_commands=REQUIRED <= {r["name"] for r in receipts if r["passed"]},
        owned_children=all(r["passed"] for r in receipts),
        new_statement_coverage=work.get("coverage_complete", False),
    )
    ready = all(gates.values())
    owned_failed = any(not r["passed"] for r in receipts)
    kind = (
        "null"
        if ready
        else "blocked"
        if evidence["failures"] and not owned_failed
        else "disqualified"
    )
    raw = Path(work["raw"])
    value: Json = dict(
        experiment_id=8566,
        task_id=e.TASK,
        milestone="2026.10.736",
        run_date="20261016",
        honest_verdict="complete_" + kind + "_v736_count_semantics_controls",
        verdict_class=kind,
        gate_check_summary=[
            *evidence["failures"],
            *[
                dict(
                    check=k,
                    upstream=e.TASK,
                    path=str(raw / "work.json"),
                    sha256=None,
                    field=k,
                    operator="==",
                    expected=True,
                    observed=v,
                    passed=False,
                )
                for k, v in gates.items()
                if not v
            ],
        ],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        historical_model_provenance=evidence.get("historical_model_provenance", []),
        rows=classifications,
        intended_count=len(classifications),
        completed_count=len(classifications),
        failed_count=sum(not r["expectation_met"] for r in classifications),
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            unit="frozen regression control",
            intended=len(classifications),
            independent_units=0,
            repeated_measurements_do_not_multiply_evidence=True,
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed original artifacts and constructed validator controls",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=ready,
        flagged_adversarial=False,
        acceptance_gates=gates,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=[],
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        phase_spans=work["phases"],
        random_seed=7368566,
        source_artifact_hashes=evidence["sources"],
        code_config_hashes=work["code"],
        raw_shard_hashes=[reference(raw / "corpus.json"), reference(raw / "reductions.json")],
        cited_upstream_artifacts=evidence["sources"],
        count_controls_ready_score=int(ready),
        control_corpus_reference=reference(raw / "corpus.json"),
        expected_classification_rows=classifications,
        reproduced_findings=evidence["reproduced_findings"],
        validator_visible_original_candidate=evidence.get("private_original"),
        original_distinct_metrics=evidence.get("metrics"),
        cardinality_reductions=evidence["cardinality_reductions"],
        count_evidence_contract=e.CONTRACT,
        owned_controller_failures=evidence["owned_controller_failures"],
        historical_true_positive_controls=evidence["historical_controls"],
        primitive_reference=reference(raw / "work.json"),
        original_learning_node_authority=evidence.get("original_node_ids", []),
        scoped_node_union=work.get("node_union", {}),
        historical_exp8555_verdict=evidence.get("historical_verdict"),
        output_path=str(output.absolute()),
        red_regression_reference=work.get("red_reference"),
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8567, 8579)
        ],
        methodology_note="Validator bug study only. Independently reduce sealed data with exact integers and units. The unchanged validator still emits seven original critical flags. Controller failures independently keep Exp8555 disqualified. No model, learner run, benefit measurement, quarantine clearing or shared validator change.",
    )
    value["field_principles"] = {
        k: "Bind invocation evidence; regression qualification cannot establish scientific benefit."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["field_principles"].update(
        count_controls_ready_score="One only when primitive reductions, sealed expectations and owned validation all pass.",
        owned_controller_failures="Independent historical defects survive any count false-positive diagnosis.",
        count_evidence_contract="Exact typed cardinalities require independent bounded data reductions; continuous identities stay critical.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def projection(evidence: Json) -> Json:
    """Ignore copy locations while retaining every scientific value and original defect."""
    projected = {
        key: evidence.get(key)
        for key in (
            "metrics",
            "reproduced_findings",
            "owned_controller_failures",
            "historical_verdict",
            "original_node_ids",
            "historical_model_provenance",
        )
    }
    projected["cardinality_reductions"] = [
        {k: v for k, v in row.items() if k != "table"} for row in evidence["cardinality_reductions"]
    ]
    projected["historical_controls"] = [
        {k: v for k, v in row.items() if k != "copy"} for row in evidence["historical_controls"]
    ]
    return projected


def replay(path: Path) -> bool:
    """Rebuild from original pins so self-consistently rehashed summaries still fail."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        if value["reproducibility_checksum"] != checksum(value) or work["corpus"] != e.corpus():
            return False
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            read(ref) if Path(ref["path"]).suffix == ".json" else reference(Path(ref["path"]))
            if reference(Path(ref["path"])) != ref:
                return False
        for receipt in work["receipts"]:
            if receipt != json.loads(
                Path(receipt["log_path"]).with_suffix(".receipt.json").read_bytes()
            ):
                return False
            for stream in ("stdout", "stderr"):
                if (
                    reference(Path(receipt[stream + "_path"]))["sha256"]
                    != receipt[stream + "_sha256"]
                ):
                    return False
        with TemporaryDirectory(prefix="carnot-8566-replay-", dir="/var/tmp") as scratch:
            reconstructed = e.measure(e.ROOT, Path(scratch))
        if projection(reconstructed) != projection(work["evidence"]):
            return False
        original_rows = read(work["evidence"]["events"])
        if e.reduce(original_rows) != reconstructed["metrics"]:
            return False
        return bool(value == build(work, Path(value["output_path"])))
    except (OSError, ValueError, KeyError, TypeError):
        return False
