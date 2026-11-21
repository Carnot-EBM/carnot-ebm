"""REQ-REPORT-8999: retain original authority without repeating scientific work."""

from collections import Counter
import json
from pathlib import Path
from typing import Any

from carnot.reporting import arc_supervisor_frontier_8720 as arc
from carnot.reporting import v768_capstone as capstone
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v755_branch_contract import reference
from carnot.reporting.v766_branch_evidence import arc_support

Json = dict[str, Any]
ROOT = capstone.ROOT
ARCHIVE = "openspec/change-proposals/research-roadmap-v768-preserved-20261111.md"
QUEUE = "openspec/change-proposals/research-roadmap-v768-queue-preserved-20261111.yaml"
CAPSTONE = "results/experiment_8998_v768_capstone.json"
BRANCH = "results/experiment_8985_v768_branch_contract.json"
GRAPH = "results/experiment_8987_v768_spline_development_admission.json"
ONLINE = "results/experiment_8988_v768_online_spline_methods.json"
PROTOCOL = "openspec/change-proposals/v768-spline-protocol.json"
ONLINE_PROTOCOL = "openspec/change-proposals/v768-online-spline-protocol.json"
LEDGER = "ops/arc_supervisor_refinement_ledger.json"
PINS = {
    ARCHIVE: "sha256:ec458dcf6743619bd1d56ddb8a97a0bd9ceb30b35e505f8372320c4b157db45b",
    QUEUE: "sha256:71f098ddce1ab271cdec4f674fe8cb7e384a66b6e4d79fff3860dc628c8c01de",
    CAPSTONE: "sha256:81688e87354e6d7fd97496776cf084a24362e44516a6f76d969f1b07f8b1c61a",
    PROTOCOL: "sha256:dd0eb38a797fd3e1ddbe172b27c4b504ac8229251eed6cce73bd3ce6114f251b",
    ONLINE_PROTOCOL: "sha256:b5808137993fc5d1e6d1e7aed19652bfe8813fefd07146818d20cedbd68734e8",
}
PINS.update(
    {
        name: capstone.PINS[n]
        for n, name in {
            8985: BRANCH,
            8986: "results/experiment_8986_v768_spline_basis_methods.json",
            8987: GRAPH,
            8988: ONLINE,
            8989: "results/experiment_8989_v768_natural_annotation_authority.json",
            8990: "results/experiment_8990_v768_graph_corpus_a.json",
            8992: "results/experiment_8992_corpus_admission.json",
            8997: "results/experiment_8997_arc_transfer_evidence.json",
        }.items()
    }
)


def check(path: Path, field: str, expected: Any, observed: Any, scope: str) -> Json:
    """Exact operands make absence distinguishable from a measured zero."""
    return dict(
        reference(path),
        check=scope,
        upstream=path.stem,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=type(observed) is type(expected) and observed == expected,
    )


def pinned(root: Path, name: str, checks: list[Json]) -> Json:
    """An independent byte pin prevents coordinated edits from becoming history."""
    path = root / name
    row = check(
        path, "sha256", PINS[name], reference(path)["sha256"] or "missing_file", "input_bytes"
    )
    checks.append(row)
    return json.loads(path.read_bytes()) if row["passed"] and path.suffix == ".json" else {}


def operands(root: Path) -> list[Json]:
    """Discover only references needed to authenticate states and retained reductions."""
    refs: list[Json] = []
    for name in [GRAPH, ONLINE, BRANCH]:
        value = pinned(root, name, [])
        refs += value.get("feature_references", [])
        refs += value.get("raw_shard_hashes", [])[:1]
        if value.get("trained_state_reference"):
            refs.append(value["trained_state_reference"])
        if name == BRANCH:
            refs += [
                r
                for r in value.get("raw_shard_hashes", [])
                if r.get("source", {}).get("path") == str(ROOT / LEDGER)
            ]
    return refs


def science(root: Path) -> Json:
    """Authenticate frozen science; these gates never require another fit or gate run."""
    checks: list[Json] = []
    graph = pinned(root, GRAPH, checks)
    online = pinned(root, ONLINE, checks)
    protocol = pinned(root, PROTOCOL, checks)
    causal = pinned(root, ONLINE_PROTOCOL, checks)
    counts: Json = {}
    if graph:
        for field in ["fitted_states_ready_score", "collection_ready_score"]:
            checks.append(check(root / GRAPH, field, 1, graph.get(field, "missing_field"), "graph"))
        admission = graph["development_admission_rows"]
        checks.append(
            check(
                root / GRAPH,
                "four_passed_admission_criteria",
                True,
                len(admission) == 4 and all(r["passed"] is True for r in admission),
                "graph",
            )
        )
        roles, graphs = graph["role_manifest"], graph["graph_manifest"]
        counts = dict(Counter(r["role"] for r in roles))
        checks.append(
            check(
                root / GRAPH,
                "role_counts",
                dict(calibration=128, test=128, online=768),
                counts,
                "graph",
            )
        )
        checks.append(
            check(
                root / GRAPH,
                "unique_graph_ids",
                1024,
                len({g["graph_id"] for g in graphs}),
                "graph",
            )
        )
        for field, rows in [("graph_manifest_hash", graphs), ("role_manifest_hash", roles)]:
            checks.append(
                check(
                    root / GRAPH,
                    field,
                    graph["source_manifest"][field],
                    canonical_hash(rows),
                    "graph",
                )
            )
        checks.append(
            check(
                root / GRAPH,
                "seven_bound_contract",
                protocol.get("seven_bound_contract"),
                graph["seven_bound_contract"],
                "graph",
            )
        )
        for ref in [
            graph["trained_state_reference"],
            *graph["feature_references"],
            graph["raw_shard_hashes"][0],
        ]:
            path = root / Path(ref["path"]).relative_to(ROOT)
            checks.append(
                check(
                    path,
                    "sha256",
                    ref["sha256"],
                    reference(path)["sha256"] or "missing_file",
                    "graph",
                )
            )
    if online:
        checks.append(
            check(
                root / ONLINE,
                "online_methods_ready_score",
                1,
                online.get("online_methods_ready_score", "missing_field"),
                "online",
            )
        )
        checks.append(
            check(
                root / ONLINE,
                "ten_bound_contract",
                causal.get("ten_bound_contract"),
                online["ten_bound_contract"],
                "online",
            )
        )
        checks.append(
            check(
                root / ONLINE_PROTOCOL,
                "five_arms",
                5,
                len(causal.get("arm_definitions", {})),
                "online",
            )
        )
    graph_ready = bool(
        graph
        and protocol
        and all(
            r["passed"]
            for r in checks
            if r["check"] == "graph" or r["path"].endswith((GRAPH, PROTOCOL))
        )
    )
    online_ready = bool(
        online
        and causal
        and all(
            r["passed"]
            for r in checks
            if r["check"] == "online" or r["path"].endswith((ONLINE, ONLINE_PROTOCOL))
        )
    )
    return dict(
        frozen_graph_authority_ready_score=int(graph_ready),
        online_protocol_ready_score=int(online_ready),
        checks=checks,
        role_counts=counts,
        development_admission_rows=graph.get("development_admission_rows", []),
        seven_bound_contract=graph.get("seven_bound_contract"),
        ten_bound_contract=online.get("ten_bound_contract"),
    )


def frontier(root: Path) -> Json:
    """Compare retained V768 identities with one saved live ledger observation."""
    checks: list[Json] = []
    rows: list[Json] = []
    before: Json = {}
    after: Json = {}
    branch = pinned(root, BRANCH, checks)
    refs: list[Json] = []
    try:
        held_ref = next(
            r
            for r in branch["raw_shard_hashes"]
            if r.get("source", {}).get("path") == str(ROOT / LEDGER)
        )
        held_path = root / Path(held_ref["path"]).relative_to(ROOT)
        held = arc.read(dict(path=str(held_path), sha256=held_ref["sha256"]))
        before = branch["arc_frontier"]["frontier_after"]
        if sorted(held["entries"]) != before["ledger_receipt_identities"]:
            raise ValueError("v768_frontier_identity_mismatch")
        ledger_path = root / LEDGER
        live = json.loads(ledger_path.read_bytes())
        closed, refs = arc.authenticate()
        rows, after = arc.delta(held, live, closed)
        refs += [reference(held_path), reference(ledger_path)]
    except (OSError, ValueError, KeyError, StopIteration) as error:
        checks.append(
            check(root / LEDGER, "new_native_receipts", "authenticated", str(error), "optional_arc")
        )
    support = arc_support(rows)
    return dict(
        ready=not any(not r["passed"] for r in checks) and support["ready"],
        checks=checks,
        references=refs,
        frontier_before=before,
        frontier_after=after,
        new_event_rows=rows,
        support=support,
        cheap_skip_count=int(not support["ready"]),
        panel_rerun_count=0,
    )
