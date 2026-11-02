"""REQ-REPORT-8567: preserve sealed authority while qualifying a validator repair.

The study compares the frozen old checker with current code on identical data.
It cannot promote a historical producer or establish a learning benefit.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
from typing import Any

from carnot.reporting import count_semantics_8566 as sealed
from carnot.reporting.count_semantics_evidence_8566 import read, reference
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash
from carnot.reporting.primary_publication import read_bound_sidecar
from scripts import adversarial_verify as av

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8567_v736_count_validator_correction"
TASK = "exp8567-count-validator-correction"
MODEL_SPECS: list[Json] = []
BASE = ROOT / "results/raw" / NAME
UPSTREAM = ROOT / "results/experiment_8566_v736_count_semantics_controls.json"
PIN = "sha256:ef6064315aee6e7f2ad32fff2d4163075ce852d54b82236ae9b342480aa4041b"
OLD_PIN = "sha256:16ad1fa750f505182dd996fdd500663d0c777a55a4cd080f377db9ff18cec43c"
REQUIRED = {
    "collection",
    "owned",
    "affected",
    "e2e018",
    "e2e021",
    "ruff_check",
    "ruff_format",
    "mypy",
    "spec",
    "cold",
    "missing",
    "owned_failure",
    "rehashed",
    "adversarial",
    "row_lint",
    "coverage",
    "combine",
    "erasure",
}


def authenticate() -> tuple[Json, Json, Json]:
    """Pin original authority instead of accepting a repaired self-reported envelope."""
    upstream = read(dict(path=str(UPSTREAM), sha256=PIN))
    work = read(upstream["primitive_reference"])
    corpus = read(upstream["control_corpus_reference"])
    old_ref = next(
        r
        for r in upstream["code_config_hashes"]
        if r["path"].endswith("/scripts/adversarial_verify.py")
    )
    if (
        old_ref["sha256"] != OLD_PIN
        or upstream["count_controls_ready_score"] != 1
        or upstream["task_id"] != "exp8566-count-semantics-controls"
        or corpus != sealed.corpus()
    ):
        raise ValueError("sealed_authority")
    if reference(BASE / "old_adversarial_verify.py")["sha256"] != OLD_PIN:
        raise ValueError("old_validator_hash")
    terminal = read(reference(Path(upstream["terminal_validation_sidecar_path"])))
    if not read_bound_sidecar(UPSTREAM, Path(terminal["publication"]["sidecar_path"]))["report"][
        "passed"
    ]:
        raise ValueError("upstream_publication")
    for ref in [
        *work["evidence"]["sources"],
        *[r for r in upstream["code_config_hashes"] if r["sha256"] != OLD_PIN],
    ]:
        if reference(Path(ref["path"])) != ref:
            raise ValueError("original_source_hash")
    return upstream, work, corpus


def candidate(corpus: Json, case: Json) -> Json:
    """Keep numeric metrics unchanged even when a sealed claim has invalid types."""
    values: Json = dict(observed_alpha=case["value"], observed_beta=case["value"])
    if case["id"] == "unequal":
        values.update(observed_alpha=2, observed_beta=1)
    if "left" in case:
        left = deepcopy(case["left"])
        if "raw_numeric_json" in case:
            left["value"] = json.loads(case["raw_numeric_json"])
        values["discrete_count_proof"] = dict(
            version=corpus["contract"]["version"],
            tables=corpus["tables"],
            claims=dict(observed_alpha=left, observed_beta=case["right"]),
        )
    return values


def findings(value: Json, checker: Any) -> list[Json]:
    """Compare the same flattened metrics through both source versions."""
    flags: list[av.Flag] = []
    checker(av._normalize_principle_wrapped_fields(av._flatten_metrics(value)), flags)
    return [f.to_dict() for f in flags]


def classifications(corpus: Json) -> list[Json]:
    """Every sealed control records exact old and new diagnostics, not just scores."""
    old = runpy.run_path(str(BASE / "old_adversarial_verify.py"))["check_tautology"]
    rows = []
    for index, case in enumerate(corpus["cases"]):
        print(
            f"[exp8567] controls completed={index} pending={len(corpus['cases']) - index}",
            flush=True,
        )
        value = candidate(corpus, case)
        before, after = findings(value, old), findings(value, av.check_tautology)
        critical = any(f["severity"] == "critical" for f in after)
        expected = case["expected"]
        passed = (
            not critical and any("structural count relation" in f["detail"] for f in after)
            if expected == "valid_discrete_identity"
            else not after
            if expected == "valid_unequal"
            else critical
        )
        rows.append(
            dict(
                unit_id=case["id"],
                expected=expected,
                metrics={k: v for k, v in value.items() if k != "discrete_count_proof"},
                old_findings=before,
                new_findings=after,
                expectation_met=passed,
                status="completed",
                missing_reason=None,
            )
        )
    return rows


def preserved(upstream: Json) -> list[Json]:
    """Unrelated real bugs and all original quarantine flags remain authoritative."""
    rows = []
    for control in upstream["historical_true_positive_controls"]:
        value = read(control["source"])
        actual = findings(value, av.check_tautology)
        rows.append(
            dict(
                source=control["source"],
                old_findings=control["current_critical_findings"],
                new_findings=actual,
                unchanged=actual == control["current_critical_findings"],
            )
        )
    original = sealed.authenticate(ROOT)
    actual = findings(original, av.check_tautology)
    rows.append(
        dict(
            source=dict(
                path=str(ROOT / "results/experiment_8555_v735_canonical_update_accounting.json"),
                sha256="sha256:e690ae89a0c2219d9ebea8558fca68168c07157190bef61ebe31313d4afee2f2",
            ),
            old_findings=original["adversarial_findings"],
            new_findings=actual,
            unchanged=actual == original["adversarial_findings"],
        )
    )
    return rows


def checksum(value: Json) -> str:
    """Bind every output field while avoiding a checksum that includes itself."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def build(work: Json, output: Path) -> Json:
    """Qualification requires complete controls, exact test union and owned coverage."""
    receipts = work["receipts"]
    gates = dict(
        preconditions=work["preconditions"]["passed"],
        original_authority=not work["failures"],
        sealed_controls=bool(work["controls"])
        and all(r["expectation_met"] for r in work["controls"]),
        preserved_detection=bool(work["preserved"])
        and all(r["unchanged"] for r in work["preserved"]),
        original_seven_relations=work.get("original_count_proof", {}).get("expectation_met", False),
        required_commands=REQUIRED <= {r["name"] for r in receipts if r["passed"]},
        owned_children=all(r["passed"] for r in receipts),
        exact_node_union=work.get("union_passed", False),
        new_statement_coverage=work.get("coverage_complete", False),
    )
    ready = all(gates.values())
    kind = (
        "null"
        if ready
        else "blocked"
        if work["failures"] and gates["owned_children"]
        else "disqualified"
    )
    raw = Path(work["raw"])
    upstream = work.get("upstream", {})
    evidence = work.get("evidence", {})
    value: Json = dict(
        experiment_id=8567,
        task_id=TASK,
        milestone="2026.10.736",
        run_date="20261016",
        honest_verdict="complete_" + kind + "_v736_count_validator_correction",
        verdict_class=kind,
        gate_check_summary=[
            *work["failures"],
            *[
                dict(
                    check=k,
                    upstream=TASK,
                    path=str(raw / "work.json"),
                    sha256=reference(raw / "work.json")["sha256"],
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
        MODEL_SPECS=MODEL_SPECS,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        historical_model_provenance=evidence.get("historical_model_provenance", []),
        rows=work["controls"],
        intended_count=len(work["controls"]),
        completed_count=len(work["controls"]),
        failed_count=sum(not r["expectation_met"] for r in work["controls"]),
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            unit="sealed regression control",
            independent_units=0,
            repeated_measurements_do_not_multiply_evidence=True,
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed frozen validator controls",
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
        random_seed=7368567,
        source_artifact_hashes=work["sources"],
        code_config_hashes=work["code"],
        raw_shard_hashes=[
            reference(raw / "controls.json"),
            *([reference(raw / "coverage.json")] if (raw / "coverage.json").is_file() else []),
        ],
        newly_owned_statement_coverage_complete=work.get("coverage_complete", False),
        new_validator_statement_coverage=work.get("new_validator_coverage", {}),
        cited_upstream_artifacts=work["sources"],
        count_validator_ready_score=int(ready),
        control_corpus_reference=upstream.get("control_corpus_reference"),
        old_new_classification_rows=work["controls"],
        original_private_count_proof=work.get("original_count_proof"),
        preserved_detection_rows=work["preserved"],
        correction_source_hashes=work["code"],
        correction_diff_reference=reference(raw / "correction.diff"),
        old_validator_reference=reference(BASE / "old_adversarial_verify.py"),
        mutation_receipts=[
            r for r in receipts if r["name"] in {"missing", "owned_failure", "rehashed", "erasure"}
        ],
        historical_quarantines=dict(
            exp8530="disqualified_unchanged",
            exp8555="disqualified_unchanged",
            exp8555_owned_controller_failures=evidence.get("owned_controller_failures", []),
        ),
        primitive_reference=reference(raw / "work.json"),
        scoped_node_union=work.get("node_union", {}),
        original_learning_node_authority=evidence.get("original_node_ids", []),
        output_path=str(output.absolute()),
        future_dependencies=[
            dict(experiment_id=i, status="future_dependency_not_observation")
            for i in range(8568, 8579)
        ],
        methodology_note="CPU arithmetic over sealed controls. Exact integers, units, bounded unique rows and two reductions prove count relations. Continuous copies and inconsistent evidence remain critical. No model load or learning benefit; historical producers remain disqualified. A fresh producer must pass the corrected validator.",
    )
    value["field_principles"] = {
        k: "Bind this invocation; regression qualification cannot establish learning benefit."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["field_principles"].update(
        count_validator_ready_score="One requires complete controls and every required scoped validation.",
        old_new_classification_rows="Identical numeric inputs through exact old and new source bytes.",
        historical_quarantines="A semantic correction cannot repair historical controller failures.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild classifications from original seals, rejecting rehashed fabricated summaries."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        if value != build(work, Path(value["output_path"])):
            return False
        upstream, original, corpus = authenticate()
        if (
            work.get("original_count_proof") != original_proof(original)
            or work["controls"] != classifications(corpus)
            or work["preserved"] != preserved(upstream)
            or work["sources"] != [reference(UPSTREAM), *original["evidence"]["sources"]]
            or work["evidence"] != original["evidence"]
            or work["upstream"] != upstream
        ):
            return False
        for ref in [
            *work["sources"],
            *work["code"],
            *value["raw_shard_hashes"],
            value["correction_diff_reference"],
        ]:
            if reference(Path(ref["path"])) != ref:
                return False
        for receipt in work["receipts"]:
            stored = json.loads(Path(receipt["log_path"]).with_suffix(".receipt.json").read_bytes())
            if stored != receipt:
                return False
            for stream in ("stdout", "stderr"):
                if (
                    reference(Path(receipt[stream + "_path"]))["sha256"]
                    != receipt[stream + "_sha256"]
                ):
                    return False
        return bool(value == build(work, Path(value["output_path"])))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def original_proof(work: Json) -> Json:
    """Annotate a private historical copy without changing its metrics or quarantine."""
    evidence = work["evidence"]
    original = read(evidence["private_original"])
    rows = read(evidence["events"])
    table = dict(rows=rows)
    table_hash = canonical_hash(table)
    totals = sealed.reduce(rows)
    operations = {key: op for op, key in sealed.KEYS.items()}
    operations["intended_count"] = "rows"
    value = deepcopy(original)
    value["discrete_count_proof"] = dict(
        version="discrete-counter-proof-v1",
        tables=dict(original=table),
        claims={
            key: dict(
                value=n,
                operation=operations[key],
                unit=sealed.CONTRACT["operations"][operations[key]],
                table="original",
                table_sha256=table_hash,
            )
            for key, n in totals.items()
        },
    )
    old = runpy.run_path(str(BASE / "old_adversarial_verify.py"))["check_tautology"]
    before = findings(original, old)
    pairs = [
        [part.split("=")[0] for part in flag["detail"].split(" agree")[0].split(" and ")]
        for flag in before
    ]
    flattened = av._flatten_metrics(value)
    groups: dict[int, Json] = {}
    for left, right in pairs:
        groups.setdefault(flattened[left], {}).update(
            {left: flattened[left], right: flattened[right]}
        )
    observed = {}
    for metrics in groups.values():
        paired = dict(metrics, discrete_count_proof=value["discrete_count_proof"])
        for flag in findings(paired, av.check_tautology):
            names = frozenset(
                part.split("=")[0]
                for part in flag["detail"].split(": structural")[0].split(" and ")
            )
            observed[names] = flag
    after = [observed[frozenset(pair)] for pair in pairs]
    return dict(
        unit_id="original_seven_private_proof",
        old_findings=before,
        new_findings=after,
        metrics=totals,
        unchanged_metrics=original["metrics"] == value["metrics"],
        verdict_class=value["verdict_class"],
        expectation_met=len(before) == 7
        and len(after) == 7
        and all(f["severity"] == "warn" for f in after),
        source=evidence["private_original"],
    )
