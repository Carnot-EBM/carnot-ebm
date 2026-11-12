"""REQ-REPORT-9031-V771: collect original B identities under current qualification."""

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

from carnot.reporting import graph_corpus_a_8961 as shared
from carnot.reporting import graph_corpus_a_8990 as authority
from carnot.reporting import graph_source_8931 as source
from carnot.reporting import invocation_provenance_9000 as custody
from carnot.reporting import scoped_provenance_9014 as checks
from carnot.reporting import bfs_error_pilot_8918 as pilot
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v756_branch_qualification import verify_measured

Json = dict[str, Any]
ROOT = shared.ROOT
NAME, TASK, EXPERIMENT_ID, SHARD, SEED = (
    "experiment_9031_v771_graph_corpus_b",
    "exp9031-graph-corpus-b",
    9031,
    "B",
    771031,
)
RUN_DATE = "20261112"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/graph_corpus_b_9031.py",
    "python/carnot/reporting/graph_corpus_runner_9031.py",
    CLI,
]
TEST = "tests/python/test_graph_corpus_b_9031.py"
TESTS = [TEST]
PINS = {
    "results/experiment_9028_v771_historical_provenance_qualification.json": (
        "sha256:a2cb53fd67a42e3bb9614849c42f27fc1d694a65bdc5d508ac5943a77e774067",
        "provenance_ready_score",
    ),
    "results/experiment_9027_v771_branch_contract.json": (
        "sha256:3c233de7321f62d1286e8e94bd045da059bc1d65a33f503b637dbd904d084ff7",
        "frozen_graph_authority_ready_score",
    ),
}
INPUTS = [*authority.INPUTS, *((n, p, f, Path(n).stem) for n, (p, f) in PINS.items())]
ROSTER_PIN = "sha256:eca6bcafffce4eb8d9345de8fadecabe723559274cc41a3388926ffd8f583c21"
ROLE_PIN = "sha256:40517a21a544a86c3df4bd00f7ad4fbbef127d5639cdb4308d7c2186e293cea1"
ExperimentTemplate, raw_complete, reference = (
    shared.ExperimentTemplate,
    shared.raw_complete,
    custody.reference,
)


def authenticate(root: Path) -> Json:
    """Current gates authorize work; only the original manifest supplies identities."""
    evidence: Json = authority.authenticate(root)
    for name, (pin, field) in PINS.items():
        path = root / name
        ref = reference(path)
        evidence["references"].append(ref)
        byte_check = custody.gate(
            path, "sha256", pin, ref["sha256"] or "missing_file", "current_authority_bytes"
        )
        evidence["checks"].append(byte_check)
        if byte_check["passed"]:
            value = json.loads(path.read_bytes())
            evidence["checks"].extend(
                custody.gate(
                    path, key, expected, value.get(key, "missing_field"), "current_authority_gate"
                )
                for key, expected in [
                    (field, 1),
                    ("required_checks_passed", True),
                    ("flagged_adversarial", False),
                ]
            )
    evidence["failures"] = [r for r in evidence["checks"] if not r["passed"]]
    return evidence


def ordered_units(protocol: Json) -> list[Json]:
    """A committed roster prevents replacement graphs after seeing responses."""
    units: list[Json] = shared.ordered_units(protocol, SHARD)
    if canonical_hash(units) != ROSTER_PIN:
        raise ValueError("B_roster_drift")
    return units


def reduce(units: list[Json], rows: list[Json]) -> Json:
    """Independent graph labels remain separate from backend confidence features."""
    return shared.reduce(units, rows, SHARD)


def inventory(root: Path, units: list[Json], raw: Path) -> Json:
    """Prior matching response bytes block new calls, even without a ready primary."""
    identities = {u["unit_id"] for u in units}
    found: set[str] = set()
    refs = []
    paths = sorted((root / "results").glob("experiment_*graph_corpus_b*.json"))
    for path in paths:
        value = json.loads(path.read_bytes())
        refs.append(custody.retain(path, raw))
        found.update(
            r["unit_id"]
            for r in value.get("response_rows", value.get("rows", []))
            if r.get("status") == "completed" and r.get("unit_id") in identities
        )
    for path in sorted(
        (root / "results/raw").glob(
            "experiment_*graph_corpus_b*/invocations/*/runtime/collection/batch-*/*/response.json"
        )
    ):
        if path.parent.name in identities:
            refs.append(custody.retain(path, raw))
            found.add(path.parent.name)
    return dict(
        references=refs,
        completed_unit_ids=sorted(found),
        check=custody.gate(raw, "previous_completed_B_identities", 0, len(found), "changed_input"),
    )


def qualified(validation: Json) -> bool:
    """Only current exact-command exits and measured new statements grant acceptance."""
    scope = validation.get("scope", {})
    return checks.receipts_pass(
        dict(
            validation_scope_manifest=scope,
            validation=validation,
            frozen_ns=scope.get("frozen_ns", 0),
        )
    ) and verify_measured(validation["coverage"], OWNED)


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Source custody needs all responses; oracle labels never establish detector benefit."""
    value: Json = shared.build(work, validation, fixture, producer=sys.modules[__name__])
    runtime = work["runtime"]
    model = runtime["model_receipt"]
    gates = work["authentication"]["checks"] + work.get("extra_checks", []) + runtime["checks"]
    gates = [
        dict(r, sha256=r.get("sha256", r.get("hash")), op="==", passed=r.get("passed", False))
        for r in gates
    ]
    for row in value["rows"]:
        row.update(group_id=row["cluster_id"], exclusion_reason=row["missing_reason"])
    for row in value["response_rows"]:
        row.update(group_id=row["cluster_id"], exclusion_reason=row["missing_reason"])
    cls = value["inference_substrate_class"]
    value.update(
        run_date=RUN_DATE,
        milestone="2026.11.771",
        gate_check_summary=gates,
        claim_scope="First-response B source custody only; BFS oracle admission is circular. No detector benefit or independent generalization.",
        inference_mode="live_gpu" if cls == "model_bounded_generation" else cls,
        inference_substrate="live_llm_inference"
        if cls == "model_bounded_generation"
        else "model_loading"
        if cls == "model_load_no_generation"
        else "blocked_no_run",
        minimum_duration_s=10
        if cls == "model_bounded_generation"
        else 2
        if cls == "model_load_no_generation"
        else 0,
        trained_head_specs=[],
        first_response_rows=value["response_rows"],
        GGUF_sha256=model.get("gguf_sha256"),
        tokenizer_sha256=model.get("gguf_sha256"),
        tokenizer_hash_scope="Embedded tokenizer is bound to the complete immutable GGUF hash.",
        cuda_offload_evidence=model,
        invocation_ledger=runtime.get("ledger", []),
        generated_tokens=sum(
            r["output_tokens"] for r in runtime.get("ledger", []) if r["operation"] == "generation"
        ),
        prompt_hashes=[dict(unit_id=u["unit_id"], sha256=u["prompt_hash"]) for u in work["units"]],
        role_manifest_hash=authority.ROLE_PIN,
        B_role_manifest_hash=ROLE_PIN,
        B_roster_hash=canonical_hash(work["units"]),
        validation_scope_manifest=validation.get("scope", {}),
        repository_health=dict(
            passed=False,
            current_scoped_acceptance=value["required_checks_passed"],
            historical_failed_health_reference=reference(ROOT / next(iter(PINS))),
        ),
        repository_health_passed=False,
        summary_recomputation=dict(
            role_counts=dict(Counter(u["role"] for u in work["units"])),
            completed=sum(r["status"] == "completed" for r in value["rows"]),
            error_count=sum(r["numerator"] or 0 for r in value["rows"]),
            denominator=512,
            method="Independent shortest distances and complete first-response roster.",
        ),
        prior_B_inventory=work.get("inventory", {}),
        budget_receipts=dict(
            preflight_deadline_s=240,
            load_deadline_s=240,
            request_deadline_s=30,
            generation_wall_budget_s=2700,
            checkpoint_requests=16,
            measurement_cap_s=3300,
            reserve_checks_s=1200,
            hard_cap_s=4800,
            phase_spans=value["phase_spans"],
        ),
        cited_upstream_artifacts=[
            dict(experiment_id=int(Path(n).name.split("_")[1]), sha256=p, fields_imported=[f])
            for n, p, f, _ in INPUTS
        ],
        reproduction_commands=[
            f"PYTHONUNBUFFERED=1 .venv/bin/python -u {CLI} --date {RUN_DATE}",
            f".venv/bin/python -u {CLI} --cold-replay <primary.json>",
        ],
    )
    value["preconditions_checked"]["environment"] = work.get("environment", {})
    if value["verdict_class"] == "blocked":
        value["honest_verdict"] = "complete_blocked_" + next(
            r["check"] for r in gates if not r["passed"]
        )
    value["field_principles"].update(
        {
            k: "Current first-response custody and exact exits only; historical work remains cited upstream."
            for k in value
        }
    )
    value["schema"] = sorted(value)
    value["reproducibility_checksum"] = pilot.checksum(value)
    return value
