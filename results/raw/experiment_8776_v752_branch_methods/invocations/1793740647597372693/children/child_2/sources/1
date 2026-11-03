"""REQ-VERIFY-8776: reduce sealed custody without treating plans as scientific results."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from carnot.reporting import v752_branch_methods as a
from carnot.reporting.current_work_receipt import canonical_hash, ZERO_INVOCATION_COUNTS
from carnot.reporting.v750_archive_bound import scoped_ready

Json = dict[str, Any]
SCORES = [
    "contract_ready_score",
    "memory_protocol_ready_score",
    "runtime_repair_ready_score",
    "arc_new_outcomes_available_score",
]


def build(work: Json, validation: Json, output: Path) -> Json:
    """Only owned validation grants readiness; optional missing inputs retain their own gates."""
    checked = scoped_ready(validation)
    preconditions = work["preconditions"]
    gates = dict(
        contract=work["contract"]["ready"],
        memory_protocol=work["memory_protocol_ready"],
        preconditions=preconditions["private_scratch_writable"]
        and all(preconditions["tools"].values())
        and preconditions["projection_fits"],
        exclusion=work["exclusion_passed"],
        mutation_controls=bool(work.get("mutation_rows"))
        and all(r["passed"] for r in work.get("mutation_rows", [])),
        owned_validation=checked,
    )
    ready = all(gates.values())
    verdict = "disqualified" if not checked else "positive" if ready else "blocked"
    raw = Path(work["raw"])
    failures = [a.gate(raw / "work.json", k, v) for k, v in gates.items() if not v]
    failures += work["runtime"]["gate_check_summary"] + work["arc"]["gate_check_summary"]
    end = validation["ended_at_utc"]
    duration = (
        datetime.fromisoformat(end) - datetime.fromisoformat(work["started_at_utc"])
    ).total_seconds()
    rows = work["contract"]["rows"]
    value = dict(
        experiment_id=8776,
        task_id=a.TASK,
        milestone=a.MILESTONE,
        run_date=work["run_date"],
        invocation_id=work["invocation_id"],
        honest_verdict=f"complete_{verdict}_v752_branch_methods",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        rows=rows,
        intended_count=13,
        completed_count=sum(bool(r["matched"]) for r in rows),
        failed_count=0,
        censored_count=sum(not r["matched"] for r in rows),
        excluded_count=0,
        independent_count=0,
        verifier_is_oracle=False,
        exposure_scope="administrative custody only; planned constructed protocols are not executed science",
        independent_generalization_score=0,
        acceptance_gates=gates,
        required_checks_passed=checked,
        flagged_adversarial=False,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=preconditions,
        duration_s=duration,
        started_at_utc=work["started_at_utc"],
        ended_at_utc=end,
        phase_spans=[
            dict(
                phase="freeze_and_scoped_validation",
                start_utc=work["started_at_utc"],
                end_utc=end,
                duration_s=duration,
            )
        ],
        random_seed=7528776,
        source_artifact_hashes=work["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[a.reference(raw / "work.json"), a.reference(raw / "validation.json")],
        cited_upstream_artifacts=work["references"],
        repository_health_reference=work["repository_health_reference"],
        scoped_validation_scope=dict(
            required_suites=validation.get("required_suites", a.TESTS + a.CONSUMERS),
            exact_node_ids=validation["union"].get("node_ids", []),
            repository_health_is_dependency=False,
            original_assertions_unchanged=True,
            e2e=["E2E-018", "E2E-021"],
        ),
        contract_ready_score=int(ready),
        memory_protocol_ready_score=int(ready and work["memory_protocol_ready"]),
        runtime_repair_ready_score=int(ready and work["runtime"]["ready"]),
        arc_new_outcomes_available_score=int(ready and work["arc"]["ready"]),
        full_task_objects=[r["task_object"] for r in rows],
        canonical_tasks_sha256=a.PIN,
        branch_protocol_refs=work["branch_protocol_refs"],
        runtime_authority=work["runtime"],
        arc_frontier_reference=work["arc"]["frontier_reference"],
        arc_new_outcome_refs=work["arc"]["new_outcome_refs"],
        preserved_primary_rows=work["preserved_primary_rows"],
        contract_mutation_rows=work.get("mutation_rows", []),
        visible_task_table=work["contract"]["visible_table"],
        learning_execution_authorized=False,
        learning_execution_ready_score=0,
        memory_execution_ready_score=0,
        new_scientific_measurements=0,
        primitive_reference=a.reference(raw / "work.json"),
        validation_reference=a.reference(raw / "validation.json"),
        publication_output=str(output.absolute()),
        methodology_note="Bind thirteen full V752 task objects to an independent digest. Freeze separate evidence, delayed-template memory, capacity and complete durable-cost protocols. Revalidate original archive and output-guard assertions under private repaired result directories. Read existing runtime authority and the real ARC frontier once each. Optional absence gates only its own branch. No old learning union, model load, generation, numerical training, live game or policy change. Administrative readiness certifies no scientific benefit or learner implementation.",
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Bind current invocation and immutable operands; independent administrative, protocol, runtime and ARC predicates do not certify scientific improvement.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Reconstruct from independently sealed primitives without polling runtime or ARC again."""
    try:
        value = json.loads(path.read_bytes())
        raw = Path(value["primitive_reference"]["path"]).parent
        seal = json.loads((raw / "independent_seal.json").read_bytes())
        if seal != {k: value[k] for k in ["primitive_reference", "validation_reference"]}:
            return False
        work, validation = (
            a.load(value["primitive_reference"]),
            a.load(value["validation_reference"]),
        )
        if a.load(a.reference(raw / "frozen_inputs.json")) != work:
            return False
        refs = work["references"] + work["sources"] + validation.get("raw_references", [])
        if any(a.reference(Path(r["path"]))["sha256"] != r["sha256"] for r in refs):
            return False
        if a.authority(Path(work["root"])) != work["contract"]:
            return False
        if work["protocols"] != a.protocols([r["task_object"] for r in work["contract"]["rows"]]):
            return False
        return bool(value == build(work, validation, Path(value["publication_output"])))
    except (OSError, ValueError, KeyError, TypeError):
        return False
