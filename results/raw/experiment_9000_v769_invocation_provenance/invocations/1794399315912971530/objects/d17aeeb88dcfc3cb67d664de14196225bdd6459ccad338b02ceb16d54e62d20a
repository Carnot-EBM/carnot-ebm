"""REQ-REPORT-8990: admit only the changed convex protocol's frozen shard."""

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

from carnot.reporting import graph_corpus_a_8961 as shared
from carnot.reporting import spline_basis_methods_8986 as basis
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
ROOT = shared.ROOT
NAME, TASK, EXPERIMENT_ID, SHARD, SEED = (
    "experiment_8990_v768_graph_corpus_a",
    "exp8990-graph-corpus-a",
    8990,
    "A",
    768090,
)
RUN_DATE = "20261111"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/graph_corpus_a_8990.py",
    "python/carnot/reporting/graph_corpus_runner_8990.py",
    CLI,
]
TESTS = ["tests/python/test_graph_corpus_a_8990.py"]
INPUTS = [
    (
        "results/experiment_8987_v768_spline_development_admission.json",
        "sha256:32fbf7d2e347f56cf3baf5d92a73b7d62ab8bd68373518d0e5346a2ac485e920",
        "collection_ready_score",
        "exp8987-spline-development-admission",
    )
]
GRAPH_PIN = "sha256:7c041eba5550c11c46dcdd4d04c063be36a129187afec26d8d24689ae099aaf8"
ROLE_PIN = "sha256:df21c3af1e5691a17f10faad2990fc552bc249e20315966057e978e5eb0c7316"
ExperimentTemplate, reference = shared.ExperimentTemplate, shared.methods.reference
raw_complete, reduce = shared.raw_complete, shared.reduce


def authenticate(root: Path) -> Json:
    """Retained authority permits calls; historical success never supplies new responses."""
    path = root / INPUTS[0][0]
    evidence: Json = dict(failures=[], references=[reference(path)], protocol={}, checks=[])
    checks = evidence["checks"]
    checks.append(
        basis.gate(
            path,
            "sha256",
            INPUTS[0][1],
            reference(path)["sha256"] or "missing_file",
            "authority_bytes",
            TASK,
        )
    )
    if checks[-1]["passed"]:
        try:
            value = json.loads(path.read_bytes())
            checks.extend(
                basis.gate(
                    path,
                    k,
                    expected,
                    value.get(k, "missing_field"),
                    "collection_gate",
                    INPUTS[0][3],
                )
                for k, expected in [
                    ("experiment_id", 8987),
                    ("task_id", INPUTS[0][3]),
                    ("collection_ready_score", 1),
                    ("required_checks_passed", True),
                ]
            )
            for field, pin in [("graph_manifest", GRAPH_PIN), ("role_manifest", ROLE_PIN)]:
                checks.append(
                    basis.gate(
                        path,
                        field,
                        pin,
                        canonical_hash(value[field]),
                        "manifest_bytes",
                        INPUTS[0][3],
                    )
                )
            if value["frozen_protocol"] != json.loads((ROOT / basis.PROTOCOL).read_bytes()):
                raise ValueError("spline_protocol_drift")
            shared.verify_references(value)
            state = value["trained_state_reference"]["sha256"]
            roles = [
                dict(
                    unit_id=q["unit_id"],
                    graph_id=g["graph_id"],
                    graph_hash=g["graph_hash"],
                    query_hash=q["prompt_hash"],
                    role=g["role"],
                    shard=g["shard"],
                    stream_id=g["stream_id"],
                    stream_position=g["stream_position"],
                    trained_state_sha256=state,
                )
                for g in value["graph_manifest"]
                for q in g["questions"]
            ]
            if roles != value["role_manifest"]:
                raise ValueError("role_state_query_join")
            evidence["protocol"] = value
            evidence["references"].extend(
                value["source_artifact_hashes"]
                + value["raw_shard_hashes"]
                + value["code_config_hashes"]
            )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            shared.depth.matched.audit.Rejected,
        ) as error:
            checks.append(
                basis.gate(
                    path,
                    "producer_structure",
                    "authenticated immutable operands",
                    str(error),
                    "producer_structure",
                    INPUTS[0][3],
                )
            )
    evidence["failures"] = [r for r in checks if not r["passed"]]
    return evidence


def ordered_units(protocol: Json) -> list[Json]:
    """Keep the admitted graph and prompt bytes without another random draw."""
    units: list[Json] = shared.ordered_units(protocol)
    return units


def qualified(validation: Json) -> bool:
    """Only this adapter and its child CLI need new statement coverage."""
    return bool(shared.qualified(validation, OWNED))


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Expose source custody and actual calls without making a detector-benefit claim."""
    value: Json = shared.build(work, validation, fixture, producer=sys.modules[__name__])
    runtime, counts = work["runtime"], work["runtime"]["counts"]
    failures = work["authentication"]["failures"] + runtime["checks"]
    gates = work["authentication"]["checks"] + [
        dict(r, sha256=r.get("hash"), op="==", passed=False) for r in runtime["checks"]
    ]
    for row in value["rows"]:
        row.update(group_id=row["cluster_id"], exclusion_reason=row["missing_reason"])
    model = runtime["model_receipt"]
    value.update(
        run_date=RUN_DATE,
        inference_substrate_class="model_bounded_generation"
        if counts["generation_calls_attempted"]
        else "model_load_no_generation"
        if counts["model_loads_attempted"]
        else "no_model_load",
        milestone="2026.11.768",
        gate_check_summary=gates,
        MODEL_SPECS=[shared.source.SETTINGS["model"]],
        model_specs=[shared.source.SETTINGS["model"]],
        trained_head_specs=[],
        claim_scope="First-response shard custody only; BFS is an oracle; no detector benefit.",
        inference_mode="live_gpu"
        if runtime["live"]
        else "model_load_no_generation"
        if counts["model_loads_attempted"]
        else "no_model_load",
        first_response_rows=value["response_rows"],
        per_game_results=value["rows"],
        model_sha256=model.get("gguf_sha256"),
        tokenizer_sha256=model.get("gguf_sha256"),
        tokenizer_hash_scope="Embedded tokenizer is bound to the entire immutable GGUF hash.",
        cuda_offload_evidence=model,
        invocation_ledger=runtime.get("ledger", []),
        prompt_hashes=[dict(unit_id=u["unit_id"], sha256=u["prompt_hash"]) for u in work["units"]],
        truth_sidecar=value["label_sidecar_hash"],
        failure_taxonomy=dict(
            Counter(r["missing_reason"] or r["parser_category"] for r in runtime["rows"])
        ),
        budget_receipts=dict(
            preflight_deadline_s=240,
            load_deadline_s=240,
            request_deadline_s=30,
            generation_wall_budget_s=2700,
            measurement_cap_s=3600,
            hard_cap_s=4800,
            phase_spans=value["phase_spans"],
        ),
        cited_upstream_artifacts=[
            dict(
                experiment_id=8987,
                sha256=INPUTS[0][1],
                fields_imported=[
                    "collection_ready_score",
                    "graph_manifest",
                    "role_manifest",
                    "trained_state_reference",
                ],
            )
        ],
        reproduction_commands=[
            f"PYTHONUNBUFFERED=1 CARNOT_FORCE_LIVE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u {CLI} --date {RUN_DATE}",
            f".venv/bin/python -u {CLI} --cold-replay <primary.json>",
        ],
    )
    if value["verdict_class"] == "blocked":
        value["honest_verdict"] = "complete_blocked_" + failures[0]["check"]
    value["field_principles"].update(
        {
            k: "Actual owned first-response evidence; historical rows are not new calls."
            for k in value
        }
    )
    value["schema"] = sorted(value)
    value["reproducibility_checksum"] = shared.pilot.checksum(value)
    return value
