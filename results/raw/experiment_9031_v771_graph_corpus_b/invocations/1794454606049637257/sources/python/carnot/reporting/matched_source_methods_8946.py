"""Freeze a restricted graph population without claiming live source feasibility.

REQ-REPORT-8946; SCENARIO-REPORT-8946-CONTRACT. Oracle labels stay outside the
unchanged response-confidence scorer; synthetic controls establish methods only.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import time
from typing import Any

from carnot.reporting import calibration_methods_8937 as calibration
from carnot.reporting import depth_opportunity_8945 as audit
from carnot.reporting import graph_source_8931 as source
from carnot.reporting import v765_branch_contract as branch
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash
from carnot.reporting.graph_source_methods_8931 import CHECKS, reference as reference
from carnot.reporting.v756_branch_qualification import verify_measured

Json = dict[str, Any]
ROOT = audit.ROOT
NAME, TASK, SEED = (
    "experiment_8946_v765_matched_source_methods",
    "exp8946-matched-source-methods",
    765046,
)
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/matched_source_methods_8946.py",
    "python/carnot/reporting/matched_source_runner_8946.py",
    CLI,
]
TEST = "tests/python/test_matched_source_methods_8946.py"
INPUTS = {
    "results/experiment_8918_v763_bfs_error_pilot.json": "sha256:b2d62a2d78535ad3bf5ca9aa83c330f2490ed7d40941a7a42b18a489aafd800e",
    "results/experiment_8931_v764_graph_source_methods.json": "sha256:810b976a4d7883a3a600c9501982eb882d1feb658b0a54cb86afea00320f5b57",
    "results/experiment_8937_v764_calibration_methods.json": "sha256:3cd2616c429c3902e2e9ece4bfa57a52b5f59b7dd364161813b7fa94b9829d25",
    "results/experiment_8945_v765_depth_opportunity_audit.json": "sha256:9f5147a1abd285111b91f30aa0314a6e89c6aa009ba3ee19e525c8eed66f9b3d",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts at boundaries so bounded work remains visible to the supervisor."""
    print(f"[exp8946] phase={phase} completed={completed} pending={pending}", flush=True)


def authenticate(root: Path) -> Json:
    """Check retained producer bytes directly, never reevaluating retired task authority."""
    result: Json = dict(failures=[], references=[], old_graphs=[])
    for name, pin in INPUTS.items():
        path = root / name
        upstream = path.stem
        result["references"].append(reference(path))
        try:
            audit.require(reference(path)["sha256"], pin, path, "sha256", upstream)
            value = json.loads(path.read_text())
            audit.require(
                value.get("required_checks_passed"), True, path, "required_checks_passed", upstream
            )
            for ref in (
                value["source_artifact_hashes"]
                + value["code_config_hashes"]
                + value["raw_shard_hashes"]
            ):
                operand = Path(ref["path"])
                audit.require(
                    reference(operand)["sha256"], ref["sha256"], operand, "sha256", upstream
                )
                result["references"].append(reference(operand))
                for module in (source, calibration, calibration.b):
                    executed = Path(module.__file__)
                    if ref.get("source_path", "").endswith("/" + executed.name):
                        audit.require(
                            reference(executed)["sha256"],
                            ref["sha256"],
                            executed,
                            "executed_helper_bytes",
                            upstream,
                        )
            if "graph_manifest" in value:
                result["old_graphs"].extend(value["graph_manifest"])
        except audit.Rejected as error:
            result["failures"].append(error.receipt)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            result["failures"].append(
                dict(
                    upstream=upstream,
                    **reference(path),
                    hash=reference(path)["sha256"],
                    check="retained_structure",
                    artifact_field="producer_json",
                    expected="original complete object",
                    observed=str(error),
                    failure_kind="rejected_producer",
                )
            )
    authority = branch.authority(root)
    result["failures"].extend(
        dict(row, hash=row.get("sha256"), failure_kind=row["observed"])
        for row in authority["failures"]
    )
    result["authority_mode"] = authority["comparison_mode"]
    result["ready"] = not result["failures"]
    return result


def graph_protocol(old_graphs: list[Json]) -> Json:
    """Rejection sampling conditions the original population on equal answer opportunity.

    Roles, prompt permutations and condition orders depend only on frozen seeds.
    Exhaustion retains every completed proposal and never changes the population.
    """
    started = time.monotonic()
    seen = {source.graph_hash(g) for g in old_graphs}
    old_seeds = {g["seed"] for g in old_graphs}
    graphs: list[Json] = []
    rejected: list[Json] = []
    splits: list[Json] = []
    orders = [[2, 3]] * 25 + [[3, 2]] * 25
    random.Random(SEED).shuffle(orders)
    roles = {}
    for shard in ("A", "B"):
        roles[shard] = ["train"] * 80 + ["calibration"] * 40 + ["test"] * 40
        random.Random(f"{SEED}/{shard}/roles").shuffle(roles[shard])
    reason = None
    for index in range(370):
        for attempt in range(2000):
            if time.monotonic() - started >= 240:
                reason = "construction_time_exhausted"
                break
            seed = 2**48 + SEED * 1000000 + index * 2000 + attempt
            graph = source.candidate(seed)
            digest = source.graph_hash(graph)
            answers = {
                str(k): source.distance_nodes(graph["nodes"], graph["edges"], graph["start"], k)
                for k in (2, 3)
            }
            rejection = (
                "duplicate_graph_or_seed"
                if digest in seen or seed in old_seeds
                else "empty_answer_set"
                if not all(answers.values())
                else "unequal_answer_sets"
                if len(answers["2"]) != len(answers["3"])
                else None
            )
            if rejection is None:
                break
            rejected.append(
                dict(
                    seed=seed,
                    graph_hash=digest,
                    reason=rejection,
                    scheduled_index=index,
                    attempt=attempt,
                )
            )
            if len(rejected) % 500 == 0:
                progress("proposal_rejections", len(graphs), 370 - len(graphs))
        else:
            reason = "proposal_budget_exhausted"
        if reason:
            break
        seen.add(digest)
        shard = "control" if index < 50 else "A" if index < 210 else "B"
        role = "control" if index < 50 else roles[shard][(index - 50) % 160]
        nodes, edges = list(graph["nodes"]), list(graph["edges"])
        rng = random.Random(f"{SEED}/{seed}/prompt")
        rng.shuffle(nodes)
        rng.shuffle(edges)
        prefix = (
            "Undirected simple graph. Nodes: "
            + ", ".join(nodes)
            + ".\nEdges: "
            + ", ".join(a + "-" + b for a, b in edges)
            + ".\n"
        )
        graph_id = f"matched-graph-{index:03d}"
        questions = []
        for k in orders[index] if index < 50 else [2]:
            prompt = (
                prefix
                + f"Name a node exactly {k} hops from {graph['start']} (shortest distance). Reply with only its two-letter label."
            )
            questions.append(
                dict(
                    unit_id=f"{graph_id}-k{k}",
                    k=k,
                    expected=answers[str(k)],
                    prompt=prompt,
                    prompt_hash=canonical_hash(prompt),
                )
            )
        graphs.append(
            dict(
                graph,
                graph_id=graph_id,
                seed=seed,
                graph_hash=digest,
                shard=shard,
                role=role,
                questions=questions,
                valid_sets=answers,
            )
        )
        if shard != "control":
            splits.append(
                dict(graph_id=graph_id, seed=seed, graph_hash=digest, role=role, shard=shard)
            )
        if (index + 1) % 50 == 0:
            progress("graph_freeze", index + 1, 369 - index)
    return dict(
        random_seed=SEED,
        complete=len(graphs) == 370,
        terminal_reason=reason,
        graph_manifest=graphs,
        proposal_rejection_rows=rejected,
        split_manifest=splits,
        feature_schema=deepcopy(source.FEATURE_SCHEMA),
        frozen_calibration_protocol=calibration.manifest(),
        future_generator_settings=dict(source.SETTINGS),
        control_acceptance=dict(
            graphs=50,
            answers=100,
            all_answers=True,
            verified_labels=True,
            usable_signals=True,
            empirical_inequality="error_k3 > error_k2",
            minimum_k2_error=0.2,
            uncertainty="paired graph 95-percent interval; empirical inequality remains mandatory",
        ),
        construction_bounds=dict(proposals_per_graph=2000, total_seconds=240),
        custody=dict(
            parser="graph_source_8931.category",
            features="graph_source_8931.signals",
            runner="graph_source_8931.collect",
            labels="separate label.json after durable first response",
            missing_features="separate backend failure; never incorrect-answer labels or substitute probabilities",
            retry_count=0,
            per_request_s=30,
            shard_generation_s=1800,
            gpu=0,
            reasoning=False,
            quantization="Q4_K_M",
        ),
    )


def validate_protocol(protocol: Json, old_graphs: list[Json]) -> bool:
    """Regenerate seed-only decisions so rehashed prompt, feature or role drift fails."""
    expected = graph_protocol(old_graphs)
    if protocol.get("complete"):
        return protocol == expected
    n, rejected = len(protocol["graph_manifest"]), len(protocol["proposal_rejection_rows"])
    prefix = deepcopy(expected)
    prefix.update(
        complete=False,
        terminal_reason=protocol["terminal_reason"],
        graph_manifest=expected["graph_manifest"][:n],
        split_manifest=expected["split_manifest"][: max(0, n - 50)],
        proposal_rejection_rows=expected["proposal_rejection_rows"][:rejected],
    )
    return protocol == prefix and protocol["terminal_reason"] in {
        "construction_time_exhausted",
        "proposal_budget_exhausted",
    }


def development(protocol: Json, old_graphs: list[Json]) -> Json:
    """Reuse the qualified estimator fixtures; graph mutations test custody, not quality."""
    progress("before_calibration_development", 0, 6)
    result: Json = calibration.development()
    progress("after_calibration_development", len(result["control_rows"]), 0)
    if protocol["complete"]:
        for fault in ("duplicate_graph", "altered_feature", "role_leak"):
            changed = deepcopy(protocol)
            if fault == "duplicate_graph":
                changed["graph_manifest"][1] = deepcopy(changed["graph_manifest"][0])
            elif fault == "altered_feature":
                changed["feature_schema"]["learned_features"].append("label")
            else:
                changed["split_manifest"][0]["role"] = "diagnostic"
            result["control_rows"].append(
                dict(control=fault, passed=not validate_protocol(changed, old_graphs))
            )
    return result


def qualified(validation: Json) -> bool:
    """Read actual unit/child statement membership; percentages alone are insufficient."""
    receipts = validation["receipts"]
    return (
        bool(receipts)
        and {r["name"] for r in receipts} == CHECKS
        and all(r["passed"] and r["exit_code"] == 0 and not r["timed_out"] for r in receipts)
        and verify_measured(validation["coverage"], OWNED)
    )


def build(work: Json, result: Json, validation: Json, fixture: bool) -> Json:
    """Reduce frozen administrative slots without counting them as independent research.

    External evidence failures block; owned failures disqualify. Complete methods
    and construction exhaustion are terminal nulls because no live model ran.
    """
    protocol, evidence = work["protocol"], work["evidence"]
    failures = evidence["failures"]
    valid = qualified(validation)
    controls = bool(result) and all(r["passed"] for r in result["control_rows"])
    ready = int(
        valid and controls and protocol.get("complete", False) and not failures and not fixture
    )
    cls = (
        "disqualified"
        if not valid and not fixture
        else "blocked"
        if failures
        else "null"
        if controls or fixture
        else "disqualified"
    )
    by_index = {i: g for i, g in enumerate(protocol.get("graph_manifest", []))}
    rows = []
    for index in range(370):
        graph = by_index.get(index)
        for k in (2, 3) if index < 50 else (2,):
            completed = graph is not None
            rows.append(
                dict(
                    unit_id=f"matched-graph-{index:03d}-k{k}",
                    cluster_id=f"matched-graph-{index:03d}",
                    seed=graph["seed"] if graph else None,
                    arm="frozen_source",
                    condition=f"k{k}",
                    k=k,
                    metric="matched_source_slot_frozen",
                    value=1 if completed else None,
                    numerator=1 if completed else None,
                    denominator=1,
                    status="completed" if completed else "censored",
                    censor_reason=None
                    if completed
                    else "external_evidence"
                    if failures
                    else protocol.get("terminal_reason"),
                    missing_reason=None if completed else "source_unavailable",
                    shard=graph["shard"] if graph else None,
                    role=graph["role"] if graph else None,
                    valid_set_size=len(graph["valid_sets"][str(k)]) if graph else None,
                )
            )
    raw = Path(work["raw"])
    value: Json = dict(work["template"])
    value.update(
        experiment_id=8946,
        task_id=TASK,
        milestone="2026.11.765",
        run_date="20261109",
        random_seed=SEED,
        honest_verdict=f"complete_{cls}_matched_source_methods",
        verdict_class=cls,
        status="blocked" if cls == "blocked" else "error" if cls == "disqualified" else "success",
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        model_invoked=False,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=result.get("numerical_training_calls", 0),
        duration_s=work["ended"] - work["started"],
        phase_spans=work["phase_spans"],
        started_at=work["started_at"],
        finished_at=work["finished_at"],
        preconditions_checked=evidence,
        verifier_is_oracle=False,
        exposure_scope="frozen restricted population and synthetic calibration only; no live collection responses",
        independent_generalization_score=0,
        intended_count=len(rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=0,
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=0,
        rows=rows,
        gate_check_summary=failures,
        source_artifact_hashes=work["retained_inputs"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[
            reference(raw / name)
            for name in ("work.json", "protocol.json", "development.json", "validation.json")
        ],
        acceptance_gates=dict(
            authenticated_inputs=not failures,
            complete_source=bool(protocol.get("complete")),
            owned_checks=valid,
            owned_controls=controls,
            live_source_feasibility_evaluated=False,
            detector_benefit_evaluated=False,
            future_depth_control=protocol.get("control_acceptance"),
            future_test_benefit=calibration.manifest()["decision_rules"]["acceptance"],
        ),
        required_checks_passed=valid,
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        global_test_debt=work["global_test_debt"],
        matched_source_ready_score=ready,
        graph_manifest=protocol.get("graph_manifest", []),
        proposal_rejection_rows=protocol.get("proposal_rejection_rows", []),
        split_manifest=protocol.get("split_manifest", []),
        protocol_hash=canonical_hash(protocol),
        feature_schema=deepcopy(source.FEATURE_SCHEMA),
        frozen_calibration_protocol=calibration.manifest(),
        calibration_protocol_hash=canonical_hash(calibration.manifest()),
        future_generator_settings=protocol.get("future_generator_settings", dict(source.SETTINGS)),
        development_control_rows=result.get("control_rows", []),
        development_comparison_rows=[
            dict(r, condition="synthetic_diagnostic", censor_reason=None)
            for r in result.get("rows", [])
        ],
        fixture_diagnostics=result.get("fixture_diagnostics"),
        terminal_reason=protocol.get("terminal_reason"),
        fixture_only=fixture,
        source_population="G(n,3/(n-1)), n uniform 50..60, conditioned on equal nonempty k2/k3 sets; no generalization to unrestricted graphs",
        flagged_adversarial=False,
        generator_weights_frozen=True,
        arc_suspended=True,
        methodology="Frozen equal-opportunity shortest-distance source and unchanged Exp8937 calibration methods.",
        methodology_note="Methods readiness is neither live source feasibility nor detector benefit. Synthetic calibration and planned graphs contribute zero independent generalization evidence.",
        reproduction_commands=[
            f"PYTHONPATH=python .venv/bin/python -u {CLI} --date 20261109",
            f".venv/bin/python -u {CLI} --cold-replay <primary.json>",
        ],
    )
    value["field_principles"] = {
        k: "Current administrative protocol or synthetic control; no inherited model compute or independent quality claim."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["field_principles"]["numerical_training_calls"] = (
        "Direct synthetic development only: two fits, three heads, 200 updates each. Validation and replay repetitions are separate verification work; no inherited or live generator training."
    )
    value["reproducibility_checksum"] = calibration.checksum(value)
    return value
