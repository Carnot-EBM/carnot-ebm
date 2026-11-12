"""REQ-REPORT-9042-V772: freeze unchanged science and a bounded prospective source."""

import json
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

import networkx as nx

from carnot.reporting import relational_methods_8959 as g
from carnot.reporting import spline_basis_methods_8986 as basis
from carnot.reporting import v772_branch_evidence as historical
from carnot.reporting import v771_branch_contract as prior_contract
from carnot.reporting import fresh_seal_custody_9042 as custody
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.invocation_provenance_9000 import reference, retain, checksum

Json = dict[str, Any]
ROOT = g.ROOT
NAME, TASK, SEED = "experiment_9042_v772_fresh_seal_methods", "exp9042-fresh-seal-methods", 772042
BRANCH = "results/experiment_9041_v772_branch_contract.json"
GRAPH = "results/experiment_8987_v768_spline_development_admission.json"
ONLINE = "results/experiment_8988_v768_online_spline_methods.json"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_fresh_seal_methods_9042.py"
OWNED = [
    "python/carnot/reporting/fresh_seal_methods_9042.py",
    "python/carnot/reporting/fresh_seal_custody_9042.py",
    "python/carnot/reporting/fresh_seal_runner_9042.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
PINS = {
    name: historical.PINS[name]
    for name in [GRAPH, ONLINE, historical.A, historical.B, historical.ARCHIVE, historical.QUEUE]
}
PINS.update(
    {
        BRANCH: "sha256:1837f2e88e964d63bb25c890dc3b4bad0292df90e4690c6c28d54c803476aba2",
        basis.PROTOCOL: basis.PROTOCOL_PIN,
        historical.ONLINE_PROTOCOL: historical.PINS[historical.ONLINE_PROTOCOL],
    }
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual counts distinguish bounded computation from a silent stalled process."""
    print(f"[exp9042] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(path: Path, field: str, expected: Any, observed: Any, check: str) -> Json:
    """Typed operands distinguish absence from an observed zero or changed bytes."""
    return basis.gate(path, field, expected, observed, check, path.stem)


def authenticate(root: Path, raw: Path) -> Json:
    """Independent pins retain rejected originals while admitting only frozen method bytes."""
    result: Json = dict(checks=[], references=[], values={}, state_hashes={}, guard_ids=[])
    for name, pin in PINS.items():
        path = root / name
        observed = reference(path)["sha256"] or "missing_file"
        result["checks"].append(gate(path, "sha256", pin, observed, "input_bytes"))
        if path.is_file():
            result["references"].append(retain(path, raw))
        if observed != pin:
            continue
        if path.suffix == ".json":
            value = json.loads(path.read_bytes())
            result["values"][name] = value
            fields = (
                [("frozen_methods_ready_score", 1), ("required_checks_passed", True)]
                if name == BRANCH
                else [("fitted_states_ready_score", 1)]
                if name == GRAPH
                else []
            )
            for field, expected in fields:
                result["checks"].append(
                    gate(path, field, expected, value.get(field, "missing_field"), "methods_gate")
                )
    result["failures"] = [r for r in result["checks"] if not r["passed"]]
    if result["failures"]:
        return result
    with patch.multiple(
        prior_contract, DESIGN=historical.ARCHIVE, ACTIVE=historical.QUEUE, STAGED=historical.QUEUE
    ):
        authority = prior_contract.authority(root)
    result["checks"].append(
        gate(
            root / historical.ARCHIVE,
            "original_full_task_digest",
            True,
            authority["ready"],
            "historical_design",
        )
    )
    producer = result["values"][GRAPH]
    refs = [
        producer["trained_state_reference"],
        producer["source_manifest"]["historical_reference"],
    ]
    for ref in refs:
        path = root / Path(ref["path"]).relative_to(ROOT)
        observed = reference(path)["sha256"] or "missing_file"
        result["checks"].append(gate(path, "sha256", ref["sha256"], observed, "retained_operand"))
        if observed == ref["sha256"]:
            result["references"].append(retain(path, raw))
    result["failures"] = [r for r in result["checks"] if not r["passed"]]
    if not result["failures"]:
        state = json.loads((root / Path(refs[0]["path"]).relative_to(ROOT)).read_bytes())
        guards = sorted({r["unit_id"] for r in producer["development_metric_rows"]})
        components = dict(
            spline=[h for h in state["heads"] if h["arm"] == "spline"],
            converged_linear=[h for h in state["heads"] if h["arm"] == "converged_linear"],
            shared_logistic=state["old_shared_linear"],
            confidence_logistic=state["confidence_linear"],
            prevalence=state["prevalence"],
            encoder=state["encoders"],
            scaling=[h["scaling"] for h in state["heads"]],
            knots=result["values"][basis.PROTOCOL]["knots"],
            guard=guards,
            prompt_parser=reference(ROOT / "python/carnot/reporting/graph_source_8931.py"),
            detector_bounds=result["values"][basis.PROTOCOL]["seven_bound_contract"],
            learning_bounds=result["values"][historical.ONLINE_PROTOCOL]["ten_bound_contract"],
        )
        result.update(
            state_hashes={k: canonical_hash(v) for k, v in components.items()}, guard_ids=guards
        )
        result["checks"].append(
            gate(root / GRAPH, "original_guard_count", 64, len(guards), "development_guard")
        )
    result["failures"] = [r for r in result["checks"] if not r["passed"]]
    return result


def history(evidence: Json) -> Json:
    """The last complete inventory plus every V768 proposal covers reused V769–V771 roles."""
    producer = evidence["values"][GRAPH]
    original = producer["source_manifest"]["historical_reference"]
    saved = next(r for r in evidence["references"] if r["source_path"] == original["path"])
    value = json.loads(Path(saved["path"]).read_bytes())
    seeds = set(value["old_seeds"])
    seeds.update(r["seed"] for r in producer["construction"]["proposal_rejection_rows"])
    seeds.update(r["seed"] for r in producer["graph_manifest"])
    return dict(
        old_graphs=value["old_graphs"] + producer["graph_manifest"],
        old_seeds=sorted(seeds),
        preserved_design_digest=prior_contract.PIN,
        historical_exposure_rows=evidence["values"][BRANCH]["historical_exposure_rows"],
    )


def role(index: int) -> Json:
    """Original label-blind role and stream permutation uses only the new namespace."""
    with patch.object(g, "SEED", SEED):
        return dict(g.role(index))


def construct(
    inventory: Json, raw: Path, *, proposal_limit: int = 20000, seconds: float = 300
) -> Json:
    """Keep the matched population and exact comparator; exhaustion cannot shrink its target."""
    started, heartbeat = time.monotonic(), time.monotonic()
    buckets: dict[Any, list[Any]] = {}
    seeds = set(inventory["old_seeds"])
    indexed = 0
    originals = [*inventory["old_graphs"], *sorted(seeds)]
    progress("history_index_before", 0, len(originals))
    for original in originals:
        graph = g.source.candidate(original) if isinstance(original, int) else original
        topology = g.nx_graph(graph)
        buckets.setdefault(g.topology_key(topology), []).append(original)
        indexed += 1
        if time.monotonic() - heartbeat >= 30:
            progress("history_index", indexed, len(originals) - indexed)
            heartbeat = time.monotonic()
        if time.monotonic() - started >= seconds:
            break
    progress("history_index_after", indexed, len(originals) - indexed)
    roster: list[Json] = []
    rows: list[Json] = []
    reason = "history_time_exhausted" if indexed != len(originals) else None
    for counter in range(min(proposal_limit, 20000)):
        if reason or len(roster) == 1024:
            break
        if time.monotonic() - started >= seconds:
            reason = "construction_time_exhausted"
            break
        seed = 2**52 + SEED * 10000000 + counter
        graph = g.source.candidate(seed)
        topology = g.nx_graph(graph)
        key = g.topology_key(topology)
        answers = [
            g.source.distance_nodes(graph["nodes"], graph["edges"], graph["start"], k)
            for k in (2, 3)
        ]
        collision = seed in seeds
        for old in buckets.get(key, []):
            previous = g.source.candidate(old) if isinstance(old, int) else old
            if nx.is_isomorphic(topology, g.nx_graph(previous)):
                collision = True
                break
        rejected = (
            "isomorphic_relabeling"
            if collision
            else "unmatched_cardinality"
            if not answers[0] or len(answers[0]) != len(answers[1])
            else None
        )
        rows.append(
            dict(
                proposal_counter=counter,
                seed=seed,
                graph_hash=g.source.graph_hash(graph),
                reason=rejected,
                roster_index=len(roster),
                status="rejected" if rejected else "accepted",
            )
        )
        seeds.add(seed)
        buckets.setdefault(key, []).append(seed)
        if rejected is None:
            with patch.object(g, "SEED", SEED):
                fresh = g.fresh_graph(graph, seed, len(roster))
            uid = f"v772-graph-{len(roster):04d}-k2"
            fresh.update(graph_id=uid.removesuffix("-k2"), unit_id=uid)
            fresh["questions"][0]["unit_id"] = uid
            roster.append(fresh)
        if time.monotonic() - heartbeat >= 30:
            progress("graph_proposals", len(rows), 20000 - len(rows))
            heartbeat = time.monotonic()
    complete = len(roster) == 1024
    result = dict(
        graph_manifest=roster,
        proposal_rows=rows,
        proposal_count=len(rows),
        complete=complete,
        terminal_reason=None if complete else reason or "proposal_budget_exhausted",
        history_indexed=indexed,
        history_pending=len(originals) - indexed,
        construction_duration_s=time.monotonic() - started,
    )
    atomic_json(raw / "construction.json", result)
    progress("graph_construction_after", len(roster), 1024 - len(roster))
    return result


def manifest(evidence: Json, result: Json, inventory_ref: Json) -> Json:
    """One content identity travels unchanged to collectors, admission and prediction tasks."""
    return dict(
        schema="v772.fresh-seal.v1",
        random_seed=SEED,
        fixture_only=False,
        graph_manifest=result["graph_manifest"],
        intended_count=1024,
        complete=result["complete"],
        per_shard=dict(calibration=64, test=64, online=384),
        state_hashes=evidence["state_hashes"],
        state_references=[
            r for r in evidence["references"] if r["source_path"].endswith("states.json")
        ],
        historical_inventory_reference=inventory_ref,
        original_guard_ids=evidence["guard_ids"],
        generator_settings=dict(g.source.SETTINGS),
        first_response_budget=1,
        prompt_parser_reference=reference(ROOT / "python/carnot/reporting/graph_source_8931.py"),
        label_access_policy=custody.POLICY,
        expected_seal_fields=custody.SEAL_FIELDS,
        detector_contract=evidence["values"][basis.PROTOCOL]["seven_bound_contract"],
        online_contract=evidence["values"][historical.ONLINE_PROTOCOL],
        consumers=[
            "exp9045-fresh-corpus-a",
            "exp9046-fresh-corpus-b",
            "exp9047-sealed-corpus-admission",
            "exp9048-sealed-calibrated-decisions",
            "exp9049-sealed-online-streams",
        ],
        historical_model_receipts="References to Exp9030/9031 and Exp8987 only; no current call credit",
    )


def build(work: Json, validation: Json, accepted: bool) -> Json:
    """Frozen identity readiness is administrative; unavailable population remains terminal blocked."""
    evidence, result = work["evidence"], work["result"]
    checks = [*evidence["checks"], *work["environment"]["checks"]]
    checks.append(
        gate(
            Path(work["raw"]) / "construction.json",
            "accepted_graph_count",
            1024,
            len(result.get("graph_manifest", [])),
            "fresh_roster_complete",
        )
    )
    failures = [r for r in checks if not r["passed"]]
    controls = work["controls"]
    qualified = accepted and not work["fixture_only"] and all(r["passed"] for r in controls)
    cls = (
        "blocked"
        if failures and work["fixture_only"]
        else "disqualified"
        if not qualified
        else "blocked"
        if failures
        else "null"
    )
    ready = int(qualified and not failures)
    count = len(result.get("graph_manifest", []))
    rows = [
        dict(
            unit_id=f"v772-graph-{i:04d}-k2",
            group_id=f"v772-graph-{i:04d}",
            arm="shared_five_arm_roster",
            condition="prospective_identity_only",
            metric="graph_role_frozen",
            value=1 if i < count else None,
            numerator=1 if i < count else None,
            denominator=1,
            status="completed" if i < count else "censored",
            exclusion_reason=None if i < count else "incomplete_population",
            outcome_status="withheld",
        )
        for i in range(1024)
    ]
    value: Json = dict(
        experiment_id=9042,
        task_id=TASK,
        milestone="2026.11.772",
        run_date="20261112",
        random_seed=SEED,
        honest_verdict="complete_disqualified_owned_validation"
        if cls == "disqualified"
        else "complete_blocked_" + failures[0]["check"]
        if failures
        else "complete_null_fresh_seal_methods",
        verdict_class=cls,
        status="error" if cls == "disqualified" else "blocked" if failures else "success",
        fresh_source_ready_score=ready,
        required_checks_passed=qualified,
        flagged_adversarial=False,
        verifier_is_oracle=False,
        independent_generalization_score=0,
        claim_scope="Prospective graph custody only; no verifier benefit or oracle-distinct generalization claim.",
        exposure_scope="Historical A TEST64 exposed and B512 unknown remain ineligible; fresh responses and truth unmeasured.",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[
            dict(arm=a, state_sha256=evidence["state_hashes"].get(a), current_fit_calls=0)
            for a in basis.ARMS
        ],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        model_invoked=False,
        numerical_training_calls=0,
        rows=rows,
        intended_count=1024,
        completed_count=count,
        independent_count=0,
        excluded_count=0,
        censored_count=1024 - count,
        failed_count=0,
        gate_check_summary=checks,
        preconditions_checked=work["environment"],
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        validation_scope_manifest=validation["scope"],
        repository_health=dict(
            passed=False, historical_reference=reference(ROOT / BRANCH), full_suite_repeated=False
        ),
        repository_health_passed=False,
        fresh_role_manifest=work["manifest_reference"],
        historical_overlap_rows=work["overlap_rows"],
        custody_mutation_rows=controls,
        state_hashes=evidence["state_hashes"],
        label_access_policy=custody.POLICY,
        expected_seal_fields=custody.SEAL_FIELDS,
        construction={
            k: v for k, v in result.items() if k not in {"graph_manifest", "proposal_rows"}
        },
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        source_artifact_hashes=evidence["references"],
        raw_shard_hashes=work["raw_references"],
        code_config_hashes=validation["scope"]["code_test_hashes"],
        cited_upstream_artifacts=[
            dict(
                experiment_id=int(Path(p).stem.split("_")[1]),
                sha256=pin,
                fields_imported=["frozen_methods_ready_score"]
                if p == BRANCH
                else ["historical_custody_and_frozen_protocol"],
            )
            for p, pin in PINS.items()
            if p.startswith("results/")
        ],
        acceptance_gates=dict(
            authenticated_inputs=not failures,
            complete_roster=result.get("complete", False),
            custody_controls=all(r["passed"] for r in controls),
            owned_checks=accepted,
        ),
        methodology_note="Authenticate original convex states and64 exposed guard identities without refitting. Enumerate original matched graph proposals with seed772042, cap20000/300s, compare exact unmarked graph isomorphism against every retained construction seed. Keep first-response intent durable, censor lost in-flight answers and deny TEST truth until both prediction seals exist. Private transport fixtures qualify mechanics only; no response, outcome or scientific reduction is measured.",
    )
    value["field_principles"] = {
        k: "Immutable primitive bytes qualify scoped administrative custody only; withheld outcomes are not zeros."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value
