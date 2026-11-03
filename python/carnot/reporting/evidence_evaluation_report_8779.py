"""REQ-VERIFY-8779: derive terminal fields from sealed measurements and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import evidence_set_evaluation_8779 as e
from carnot.reporting import evidence_set_methods_8768 as original
from carnot.reporting.evidence_qualification_report_8777 import audit_validation
from carnot.reporting.current_work_receipt import canonical_hash, ZERO_INVOCATION_COUNTS
from carnot.reporting.v746_retained_evidence import load, reference

Json = dict[str, Any]


def build(work: Json, validation: Json, output: Path) -> Json:
    """Execution readiness and scientific benefit have different gates and separate meanings."""
    frozen = load(work["frozen_reference"])
    inputs, labels = frozen["inputs"], frozen["labels"]
    decisions = load(work["decisions_reference"])["rows"]
    rows = e.score(inputs, labels, decisions)
    summary = e.summarize(rows)
    checks = bool(validation["receipts"]) and all(r["passed"] for r in validation["receipts"])
    checks = checks and validation["union"]["passed"] and validation["coverage"]["complete"]
    gates = dict(
        preconditions=work["preconditions_checked"]["passed"],
        upstream_authenticated=frozen["passed"],
        complete_rows=len(rows) == 8640,
        decision_audit=e.audit(inputs, decisions),
        truth_controls=all(
            summary[k].get("truth_controls_passed", False)
            for k in ["exposed_replication_metrics", "new_family_metrics"]
        ),
        scoped_validation=bool(checks),
    )
    ready = all(gates.values())
    verdict = (
        "blocked"
        if not gates["upstream_authenticated"]
        else "disqualified"
        if not ready
        else "circular_positive"
        if summary["scientific_promotion"]
        else "null"
    )
    diagnostics = list(frozen["diagnostics"])
    diagnostics += [
        dict(
            upstream=e.TASK,
            **work["frozen_reference"],
            check=k,
            field=k,
            operator="==",
            expected=True,
            observed=v,
        )
        for k, v in gates.items()
        if not v
    ]
    value: Json = dict(
        experiment_id=8779,
        task_id=e.TASK,
        milestone="2026.11.752",
        run_date=work["run_date"],
        invocation_id=work["invocation_id"],
        honest_verdict="complete_" + verdict + "_v752_evidence_set_evaluation",
        verdict_class=verdict,
        gate_check_summary=diagnostics,
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        rows=rows,
        intended_count=8640,
        completed_count=len(rows),
        failed_count=0,
        censored_count=8640 - len(rows),
        excluded_count=0,
        independent_count=len({r["world_id"] for r in rows if r["split"] == "evaluation"}),
        independent_count_scope="120 six-edge evaluation worlds;120 exposed five-edge replication worlds separate; paired surfaces never increase N",
        verifier_is_oracle=False,
        oracle_dependent_benchmark=True,
        exposure_scope="V751 is exposed replication. Six-edge graphs were frozen by Exp8777 and are unseen by the unchanged V751 rule, but qualification and this implementation inspected them. No blinded or natural-data generalization claim.",
        independent_generalization_score=0,
        acceptance_gates=gates,
        required_checks_passed=bool(checks),
        flagged_adversarial=verdict == "disqualified",
        validation_receipts=validation["receipts"],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        phase_spans=work["phase_spans"],
        random_seed=e.SEED,
        source_artifact_hashes=frozen["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=work["shards"] + validation.get("raw_references", []),
        cited_upstream_artifacts=frozen["references"],
        repository_health_reference=work["repository_health_reference"],
        scoped_validation_scope=dict(
            owned=e.OWNED,
            tests=e.TESTS,
            consumers=original.CONSUMERS,
            named_e2es=["E2E-018 private CLI", "E2E-021 private CLI"],
            full_suite_pass_claimed=False,
        ),
        evidence_rows_ready_score=int(ready),
        owned_statement_coverage=validation["coverage"],
        retained_child_custody=validation["union"],
        equal_call_ceiling=5,
        frozen_method_reference=work["frozen_reference"],
        primitive_reference=reference(Path(work["raw"]) / "work.json"),
        validation_reference=reference(Path(work["raw"]) / "validation.json"),
        frozen_input_reference=work["frozen_reference"],
        publication_output=str(output.absolute()),
        methodology_note="Frozen signed-record sufficiency evaluation: six truth interventions, two paired surfaces, three arms and equal five-call ceilings. Labels are independently enumerated Boolean completions, sealed before input-only measurement. Exposed replication and six-edge evaluation strata use separate10000-draw world-cluster bootstraps, Bonferroni across two primary contrasts, simultaneous97.5 percent intervals. Benefit requires both lower intervals above zero and no worse false acceptance or valid rejection. The historical joint_symbolic control accepts positive literals even under explicit conflict; its semantics remain frozen. Oracle-dependent gains are circular_positive and do not establish natural hallucination detection. No model or training executes.",
        **summary,
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Exact task and invocation bind sealed operands; readiness measures execution, not natural usefulness.",
    )
    value["field_principles"].update(
        evidence_rows_ready_score="One only for complete authenticated rows, reductions and passing owned validation.",
        exposure_scope="Six-edge graphs are new to the frozen V751 rule, already inspected during qualification; constructed oracle truth is not natural factuality.",
        independent_count="Primary N=120 six-edge worlds;120 exposed worlds are separate, renamings are paired.",
        gate_check_summary="Every failed gate retains upstream,path/hash,check,field,operator,expected and observed.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Check independent seals before rebuilding labels, decisions and all terminal reductions."""
    try:
        value = json.loads(path.read_bytes())
        raw = Path(value["primitive_reference"]["path"]).parent
        keys = ["primitive_reference", "validation_reference", "frozen_input_reference"]
        if json.loads((raw / "independent_seal.json").read_bytes()) != {k: value[k] for k in keys}:
            return False
        work, validation = load(value["primitive_reference"]), load(value["validation_reference"])
        if not audit_validation(validation):
            return False
        frozen = load(work["frozen_reference"])
        if e.freeze(Path(work["upstream_path"])) != frozen:
            return False
        for ref in (
            work["sources"]
            + work["shards"]
            + frozen["references"]
            + validation.get("raw_references", [])
        ):
            if reference(Path(ref["path"])) != ref:
                return False
        decisions = load(work["decisions_reference"])["rows"]
        if not e.audit(frozen["inputs"], decisions):
            return False
        return bool(value == build(work, validation, Path(value["publication_output"])))
    except (OSError, ValueError, KeyError, TypeError):
        return False
