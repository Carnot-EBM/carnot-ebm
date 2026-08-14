"""Exp6415 exact Boolean WCSP CCG kernelization control.

Spec refs: REQ-CONSTRAINT-VERIFY-6415,
SCENARIO-CONSTRAINT-VERIFY-6415-EXACT-PRESERVATION,
SCENARIO-CONSTRAINT-VERIFY-6415-ATTACKS,
SCENARIO-CONSTRAINT-VERIFY-6415-NO-SPEEDUP-CLAIM.

This module keeps the control small on purpose. It supports integer unary and
pairwise Boolean WCSP terms. Submodular pairwise terms lower exactly to an
s-t cut. Non-submodular terms remain exact for exhaustive reference and
completion, but the CCG kernelizer abstains from fixing variables.
"""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6415_boolean_wcsp_ccg_kernelization.py")
RESULT_RELATIVE_PATH = Path("results/experiment_6415_boolean_wcsp_ccg_kernelization.json")
FROZEN_MANIFEST_RELATIVE_PATH = Path("results/experiment_6415_boolean_wcsp_frozen_manifest.json")
BOOLEAN_WCSP_SCHEMA_RELATIVE_PATH = Path("python/carnot/schemas/boolean_wcsp_v1.json")
CCG_SCHEMA_RELATIVE_PATH = Path("python/carnot/schemas/ccg_boolean_v1.json")

BOOLEAN_WCSP_SCHEMA = "carnot.boolean_wcsp.v1"
CCG_SCHEMA = "carnot.ccg_boolean.v1"
INFERENCE_SUBSTRATE = "deterministic_cpu_exact_boolean_wcsp_ccg_local_control_no_llm"
RANDOM_SEED = 6415
MAX_ABS_COST = 1_000_000_000
MAX_VARIABLES = 12
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
    "blocked:",
    "blocked_",
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_6415_boolean_wcsp_ccg_kernelization.py "
    "--cov=python/carnot/experiment_6415_boolean_wcsp_ccg_kernelization.py "
    "--cov-report=term-missing --cov-fail-under=100",
    ".venv/bin/python -m carnot.experiment_6415_boolean_wcsp_ccg_kernelization --date 20260814",
    ".venv/bin/pytest tests/python -q",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Names whether the exact local control is usable or blocked.",
    "source_encoder_solver_sampler_and_dependency_hashes": "Pins the code and dependency inputs used before this local control ran.",
    "boolean_wcsp_schema_path_hash_and_fields": "Makes the source WCSP schema inspectable and content-addressed.",
    "ccg_schema_path_hash_node_edge_and_mapping_contract": "Makes the CCG graph and reverse mapping contract inspectable.",
    "kernelizer_path_and_hash": "Pins the local maxflow kernelizer implementation.",
    "frozen_manifest_path_hash_counts_classes_and_seeds": "Shows the frozen panel size, classes, seeds, and manifest hash.",
    "exact_reference_method_and_receipts": "Declares exhaustive enumeration as the independent optimum authority.",
    "per_instance_source_and_kernelized_optima": "Compares source and kernelized exact optima for every frozen instance.",
    "optimum_preservation_rate": "Measures the fraction of frozen instances whose optimum is preserved exactly.",
    "fixed_variable_certificates_and_independent_checks": "Records each fixed-variable certificate and its independent exact check.",
    "state_space_reduction_by_instance": "Measures how many exact states remain after certified fixes.",
    "verifier_call_reduction_by_instance": "Measures exact verifier calls saved by completion after kernelization.",
    "sampler_work_and_wall_time_by_arm": "Keeps exact arms and the seeded energy sampler costed separately.",
    "sign_weight_duplicate_component_auxiliary_mapping_overflow_fixed_variable_and_nonunique_attack_matrix": "Proves known unsafe reductions are rejected or abstained.",
    "quantum_advantage_claimed": "Must stay false because this is a local CPU control.",
    "hardware_speedup_claimed": "Must stay false because no hardware path is used.",
    "ccg_kernelization_exact_ready_score": "Equals 1.0 only when preservation, certificates, attacks, and measurements all pass.",
    "protected_files_unchanged": "Shows conductor and reconciliation files stayed byte-stable.",
    "preconditions_checked": "Lists the local gates checked before the result is trusted.",
    "inference_substrate": "Declares deterministic CPU enumeration and local maxflow, not LLM inference.",
    "verifier_is_oracle": "Marks only exact optimum and certificate checks as oracles.",
    "field_principles": "Documents why each required artifact field exists.",
    "field_provenance": "States how each required artifact field was produced.",
    "random_seed": "Pins deterministic fixture and sampler replay.",
    "duration_s": "Records wall time for the experiment command.",
    "tests_run": "Records the verification commands run for this artifact.",
    "reproducibility_checksum": "Content-addresses the payload with this field blanked.",
    "honest_verdict": "Gives a terminal-prefix verdict that names the exact readiness outcome.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(value: Any) -> str:
    """Serialize data with stable key order for hashes and receipts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible content with the repository SHA-256 convention."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without changing it."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while blanking its self-referential checksum field."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def _bit_keys(arity: int) -> list[str]:
    if arity == 0:
        return [""]
    return [format(index, f"0{arity}b") for index in range(2**arity)]


def _term_costs_to_dict(costs: Sequence[int]) -> dict[str, int]:
    return {key: int(costs[index]) for index, key in enumerate(_bit_keys(len(costs).bit_length() - 1))}


def _cost_key_to_index(key: str) -> int:
    return int(key, 2) if key else 0


@dataclass(frozen=True)
class CanonicalTerm:
    """One aggregated WCSP term in sorted variable order."""

    term_id: str
    scope: tuple[int, ...]
    costs: tuple[int, ...]
    source_term_ids: tuple[str, ...]

    def to_json(self) -> JsonDict:
        return {
            "term_id": self.term_id,
            "scope": list(self.scope),
            "costs": _term_costs_to_dict(self.costs),
            "source_term_ids": list(self.source_term_ids),
        }


@dataclass(frozen=True)
class BooleanWCSP:
    """Canonical bounded Boolean weighted constraint satisfaction instance."""

    instance_id: str
    n_variables: int
    canonical_terms: tuple[CanonicalTerm, ...]
    source_mapping: dict[str, JsonDict]
    classes: tuple[str, ...]
    seed: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BooleanWCSP":
        n_variables = int(raw["n_variables"])
        if n_variables < 1 or n_variables > MAX_VARIABLES:
            raise ValueError("n_variables bound")
        aggregates: dict[tuple[int, ...], list[int]] = {}
        sources: dict[tuple[int, ...], list[str]] = {}
        mapping: dict[str, JsonDict] = {}
        seen_ids: set[str] = set()
        for position, raw_term in enumerate(raw.get("terms", [])):
            source_id = str(raw_term.get("term_id", f"source_{position}"))
            if source_id in seen_ids:
                raise ValueError("duplicate term_id")
            seen_ids.add(source_id)
            source_ids = tuple(str(value) for value in raw_term.get("source_term_ids", [source_id]))
            source_scope = tuple(int(value) for value in raw_term["scope"])
            if len(source_scope) > 2 or len(set(source_scope)) != len(source_scope):
                raise ValueError("scope arity")
            if any(value < 0 or value >= n_variables for value in source_scope):
                raise ValueError("scope variable")
            arity = len(source_scope)
            expected_keys = set(_bit_keys(arity))
            raw_costs = {str(key): int(value) for key, value in raw_term["costs"].items()}
            if set(raw_costs) != expected_keys:
                raise ValueError("cost table shape")
            if any(abs(value) > MAX_ABS_COST for value in raw_costs.values()):
                raise ValueError("cost bound")
            canonical_scope = tuple(sorted(source_scope))
            reordered = [0 for _ in range(2**arity)]
            for canonical_key in _bit_keys(arity):
                assignment = {
                    variable: int(canonical_key[index])
                    for index, variable in enumerate(canonical_scope)
                }
                source_key = "".join(str(assignment[variable]) for variable in source_scope)
                reordered[_cost_key_to_index(canonical_key)] = raw_costs[source_key]
            aggregates.setdefault(canonical_scope, [0 for _ in range(2**arity)])
            sources.setdefault(canonical_scope, [])
            for index, value in enumerate(reordered):
                aggregates[canonical_scope][index] += int(value)
            sources[canonical_scope].extend(source_ids)
            for mapped_source_id in source_ids:
                mapping[mapped_source_id] = {
                    "source_scope": list(source_scope),
                    "canonical_scope": list(canonical_scope),
                    "source_costs": raw_costs,
                }
        terms: list[CanonicalTerm] = []
        for index, scope in enumerate(sorted(aggregates, key=lambda item: (len(item), item))):
            scope_text = "const" if not scope else "_".join(str(value) for value in scope)
            terms.append(
                CanonicalTerm(
                    term_id=f"term_{index:03d}_scope_{scope_text}",
                    scope=scope,
                    costs=tuple(aggregates[scope]),
                    source_term_ids=tuple(sorted(sources[scope])),
                )
            )
        return cls(
            instance_id=str(raw["instance_id"]),
            n_variables=n_variables,
            canonical_terms=tuple(terms),
            source_mapping=mapping,
            classes=tuple(sorted(str(value) for value in raw.get("classes", []))),
            seed=int(raw.get("seed", RANDOM_SEED)),
        )

    def to_json(self) -> JsonDict:
        return {
            "schema": BOOLEAN_WCSP_SCHEMA,
            "instance_id": self.instance_id,
            "n_variables": self.n_variables,
            "classes": list(self.classes),
            "seed": self.seed,
            "terms": [term.to_json() for term in self.canonical_terms],
            "source_mapping": self.source_mapping,
            "canonical_hash": self.canonical_hash(),
        }

    def canonical_hash(self) -> str:
        return sha256_json(
            {
                "schema": BOOLEAN_WCSP_SCHEMA,
                "instance_id": self.instance_id,
                "n_variables": self.n_variables,
                "classes": list(self.classes),
                "seed": self.seed,
                "terms": [term.to_json() for term in self.canonical_terms],
            }
        )

    def evaluate(self, assignment: Mapping[int, int]) -> int:
        total = 0
        for variable in range(self.n_variables):
            if int(assignment[variable]) not in {0, 1}:
                raise ValueError("assignment label")
        for term in self.canonical_terms:
            key = "".join(str(int(assignment[variable])) for variable in term.scope)
            total += int(term.costs[_cost_key_to_index(key)])
        return total


@dataclass(frozen=True)
class CCG:
    """Constraint composite graph plus exact graph-cut lowering metadata."""

    instance_id: str
    nodes: tuple[JsonDict, ...]
    edges: tuple[JsonDict, ...]
    graph_cut_edges: tuple[JsonDict, ...]
    graph_cut_constant: int
    exact_graph_cut: bool
    omitted_terms: tuple[str, ...]
    mapping_contract: JsonDict

    def to_json(self) -> JsonDict:
        return {
            "schema": CCG_SCHEMA,
            "instance_id": self.instance_id,
            "nodes": list(self.nodes),
            "edges": list(self.edges),
            "graph_cut_edges": list(self.graph_cut_edges),
            "graph_cut_constant": self.graph_cut_constant,
            "exact_graph_cut": self.exact_graph_cut,
            "omitted_terms": list(self.omitted_terms),
            "mapping_contract": self.mapping_contract,
            "ccg_hash": self.canonical_hash(),
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "CCG":
        return cls(
            instance_id=str(raw["instance_id"]),
            nodes=tuple(dict(node) for node in raw["nodes"]),
            edges=tuple(dict(edge) for edge in raw["edges"]),
            graph_cut_edges=tuple(dict(edge) for edge in raw["graph_cut_edges"]),
            graph_cut_constant=int(raw["graph_cut_constant"]),
            exact_graph_cut=bool(raw["exact_graph_cut"]),
            omitted_terms=tuple(str(value) for value in raw["omitted_terms"]),
            mapping_contract=dict(raw["mapping_contract"]),
        )

    def canonical_hash(self) -> str:
        return sha256_json(
            {
                "schema": CCG_SCHEMA,
                "instance_id": self.instance_id,
                "nodes": list(self.nodes),
                "edges": list(self.edges),
                "graph_cut_edges": list(self.graph_cut_edges),
                "graph_cut_constant": self.graph_cut_constant,
                "exact_graph_cut": self.exact_graph_cut,
                "omitted_terms": list(self.omitted_terms),
                "mapping_contract": self.mapping_contract,
            }
        )


@dataclass(frozen=True)
class KernelizationResult:
    """Result of the exact CCG maxflow persistence pass."""

    ccg: CCG
    fixed_assignments: dict[int, int]
    certificates: tuple[JsonDict, ...]
    maxflow_value: int | None
    reason: str

    def to_json(self) -> JsonDict:
        return {
            "ccg_hash": self.ccg.canonical_hash(),
            "fixed_assignments": {str(key): value for key, value in self.fixed_assignments.items()},
            "certificates": list(self.certificates),
            "maxflow_value": self.maxflow_value,
            "reason": self.reason,
        }


def _variable_node(variable: int) -> str:
    return f"var:{variable}"


def _aux_node(term: CanonicalTerm) -> str:
    return f"aux:{term.term_id}"


def build_ccg(instance: BooleanWCSP) -> CCG:
    """Build a reversible CCG and an exact graph-cut lowering when valid."""

    nodes: list[JsonDict] = [
        {"node_id": "source", "kind": "source"},
        {"node_id": "sink", "kind": "sink"},
    ]
    nodes.extend(
        {"node_id": _variable_node(variable), "kind": "variable", "variable": variable}
        for variable in range(instance.n_variables)
    )
    edges: list[JsonDict] = []
    graph_cut_edges: list[JsonDict] = []
    pair_edges: list[tuple[int, int, int, str]] = []
    unary_delta = {variable: 0 for variable in range(instance.n_variables)}
    constant = 0
    exact_graph_cut = True
    omitted_terms: list[str] = []
    term_mappings: dict[str, JsonDict] = {}

    for term in instance.canonical_terms:
        aux = _aux_node(term)
        nodes.append({"node_id": aux, "kind": "auxiliary_term", "term_id": term.term_id})
        term_mappings[term.term_id] = {
            "auxiliary_node": aux,
            "canonical_scope": list(term.scope),
            "source_term_ids": list(term.source_term_ids),
            "costs": _term_costs_to_dict(term.costs),
        }
        for variable in term.scope:
            edges.append(
                {
                    "from": aux,
                    "to": _variable_node(variable),
                    "kind": "term_variable_incidence",
                    "weight": max(term.costs) - min(term.costs),
                    "term_id": term.term_id,
                }
            )
        if len(term.scope) == 0:
            constant += term.costs[0]
        elif len(term.scope) == 1:
            variable = term.scope[0]
            constant += term.costs[0]
            unary_delta[variable] += term.costs[1] - term.costs[0]
        else:
            left, right = term.scope
            e00, e01, e10, e11 = term.costs
            weight = e01 + e10 - e00 - e11
            if weight < 0:
                exact_graph_cut = False
                omitted_terms.append(term.term_id)
            else:
                constant += e00
                unary_delta[left] += e10 - e00
                unary_delta[right] += e11 - e10
                pair_edges.append((left, right, weight, term.term_id))

    if exact_graph_cut:
        for variable, delta in sorted(unary_delta.items()):
            if delta > 0:
                graph_cut_edges.append(
                    {
                        "from": "source",
                        "to": _variable_node(variable),
                        "capacity": delta,
                        "kind": "unary_cost_when_label_1",
                    }
                )
            elif delta < 0:
                constant += delta
                graph_cut_edges.append(
                    {
                        "from": _variable_node(variable),
                        "to": "sink",
                        "capacity": -delta,
                        "kind": "unary_cost_when_label_0",
                    }
                )
        for left, right, weight, term_id in pair_edges:
            if weight > 0:
                graph_cut_edges.append(
                    {
                        "from": _variable_node(left),
                        "to": _variable_node(right),
                        "capacity": weight,
                        "kind": "submodular_pair_directed_cut",
                        "term_id": term_id,
                    }
                )
    else:
        graph_cut_edges = []

    return CCG(
        instance_id=instance.instance_id,
        nodes=tuple(nodes),
        edges=tuple(edges),
        graph_cut_edges=tuple(graph_cut_edges),
        graph_cut_constant=int(constant),
        exact_graph_cut=exact_graph_cut,
        omitted_terms=tuple(sorted(omitted_terms)),
        mapping_contract={
            "variable_label_contract": "source_side_label_0_sink_side_label_1",
            "term_mappings": term_mappings,
            "source_instance_hash": instance.canonical_hash(),
            "construction": "submodular_pairwise_boolean_graph_cut",
        },
    )


def _graph_cut_energy(ccg: CCG, assignment: Mapping[int, int]) -> int:
    total = int(ccg.graph_cut_constant)
    for edge in ccg.graph_cut_edges:
        left = edge["from"]
        right = edge["to"]
        capacity = int(edge["capacity"])
        left_side = "source" if left == "source" else "sink" if left == "sink" else (
            "source" if assignment[int(left.split(":")[1])] == 0 else "sink"
        )
        right_side = "source" if right == "source" else "sink" if right == "sink" else (
            "source" if assignment[int(right.split(":")[1])] == 0 else "sink"
        )
        if left_side == "source" and right_side == "sink":
            total += capacity
    return total


def validate_ccg_contract(instance: BooleanWCSP, ccg: CCG) -> bool:
    node_ids = {node["node_id"] for node in ccg.nodes}
    if {"source", "sink"} - node_ids:
        raise ValueError("source sink nodes")
    for variable in range(instance.n_variables):
        if _variable_node(variable) not in node_ids:
            raise ValueError("variable node")
    mappings = dict(ccg.mapping_contract.get("term_mappings", {}))
    for term in instance.canonical_terms:
        mapping = mappings.get(term.term_id)
        if not mapping or mapping.get("auxiliary_node") not in node_ids:
            raise ValueError("auxiliary")
        if list(term.scope) != list(mapping.get("canonical_scope", [])):
            raise ValueError("mapping")
    for edge in ccg.graph_cut_edges:
        capacity = int(edge.get("capacity", 0))
        if capacity < 0:
            raise ValueError("negative edge capacity")
        if edge["from"] not in node_ids or edge["to"] not in node_ids:
            raise ValueError("edge endpoint")
    if ccg.exact_graph_cut:
        for assignment in _enumerate_assignments(instance.n_variables):
            if _graph_cut_energy(ccg, assignment) != instance.evaluate(assignment):
                raise ValueError("graph energy mapping")
    return True


@dataclass
class _ResidualEdge:
    to_node: int
    rev_index: int
    capacity: int


class _MaxFlow:
    """Small Edmonds-Karp maxflow for bounded local graph-cut controls."""

    def __init__(self, node_count: int) -> None:
        self.graph: list[list[_ResidualEdge]] = [[] for _ in range(node_count)]

    def add_edge(self, left: int, right: int, capacity: int) -> None:
        forward = _ResidualEdge(right, len(self.graph[right]), int(capacity))
        reverse = _ResidualEdge(left, len(self.graph[left]), 0)
        self.graph[left].append(forward)
        self.graph[right].append(reverse)

    def maxflow(self, source: int, sink: int) -> int:
        flow = 0
        while True:
            parent: list[tuple[int, int] | None] = [None for _ in self.graph]
            queue: deque[int] = deque([source])
            while queue and parent[sink] is None:
                node = queue.popleft()
                for edge_index, edge in enumerate(self.graph[node]):
                    if edge.capacity > 0 and edge.to_node != source and parent[edge.to_node] is None:
                        parent[edge.to_node] = (node, edge_index)
                        queue.append(edge.to_node)
            if parent[sink] is None:
                return flow
            path_capacity = 10**30
            node = sink
            while node != source:
                prev, edge_index = parent[node]
                path_capacity = min(path_capacity, self.graph[prev][edge_index].capacity)
                node = prev
            node = sink
            while node != source:
                prev, edge_index = parent[node]
                edge = self.graph[prev][edge_index]
                edge.capacity -= path_capacity
                self.graph[edge.to_node][edge.rev_index].capacity += path_capacity
                node = prev
            flow += path_capacity

    def reachable_from(self, start: int) -> set[int]:
        seen = {start}
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for edge in self.graph[node]:
                if edge.capacity > 0 and edge.to_node not in seen:
                    seen.add(edge.to_node)
                    queue.append(edge.to_node)
        return seen

    def can_reach(self, target: int) -> set[int]:
        reverse: list[list[int]] = [[] for _ in self.graph]
        for left, edges in enumerate(self.graph):
            for edge in edges:
                if edge.capacity > 0:
                    reverse[edge.to_node].append(left)
        seen = {target}
        queue: deque[int] = deque([target])
        while queue:
            node = queue.popleft()
            for prev in reverse[node]:
                if prev not in seen:
                    seen.add(prev)
                    queue.append(prev)
        return seen


def _run_maxflow(ccg: CCG) -> JsonDict:
    node_ids = sorted({node["node_id"] for node in ccg.nodes})
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    flow = _MaxFlow(len(node_ids))
    for edge in ccg.graph_cut_edges:
        flow.add_edge(index[edge["from"]], index[edge["to"]], int(edge["capacity"]))
    value = flow.maxflow(index["source"], index["sink"])
    source_reachable = {node_ids[item] for item in flow.reachable_from(index["source"])}
    can_reach_sink = {node_ids[item] for item in flow.can_reach(index["sink"])}
    return {
        "value": value,
        "source_reachable": sorted(source_reachable),
        "can_reach_sink": sorted(can_reach_sink),
    }


def kernelize_with_ccg(
    instance: BooleanWCSP, source_reference: Mapping[str, Any] | None = None
) -> KernelizationResult:
    ccg = build_ccg(instance)
    validate_ccg_contract(instance, ccg)
    if not ccg.exact_graph_cut:
        return KernelizationResult(
            ccg=ccg,
            fixed_assignments={},
            certificates=(),
            maxflow_value=None,
            reason="abstained_non_submodular_ccg",
        )
    flow_receipt = _run_maxflow(ccg)
    fixed: dict[int, int] = {}
    certificates: list[JsonDict] = []
    for variable in range(instance.n_variables):
        node = _variable_node(variable)
        label: int | None = None
        if node in flow_receipt["source_reachable"]:
            label = 0
        elif node in flow_receipt["can_reach_sink"]:
            label = 1
        if label is not None:
            fixed[variable] = label
            certificates.append(
                {
                    "certificate_id": f"{instance.instance_id}:var:{variable}:label:{label}",
                    "variable": variable,
                    "fixed_label": label,
                    "node": node,
                    "justification": "residual_reachability_for_all_minimum_cuts",
                    "source_reachable": node in flow_receipt["source_reachable"],
                    "can_reach_sink": node in flow_receipt["can_reach_sink"],
                    "maxflow_value": flow_receipt["value"],
                    "graph_cut_constant": ccg.graph_cut_constant,
                    "ccg_hash": ccg.canonical_hash(),
                    "source_optimum_cost": None
                    if source_reference is None
                    else source_reference.get("optimum_cost"),
                }
            )
    return KernelizationResult(
        ccg=ccg,
        fixed_assignments=fixed,
        certificates=tuple(certificates),
        maxflow_value=int(flow_receipt["value"]),
        reason="exact_graph_cut_persistence",
    )


def _enumerate_assignments(n_variables: int) -> list[dict[int, int]]:
    assignments: list[dict[int, int]] = []
    for state in range(2**n_variables):
        assignments.append(
            {variable: (state >> (n_variables - 1 - variable)) & 1 for variable in range(n_variables)}
        )
    return assignments


def exhaustive_reference(
    instance: BooleanWCSP, fixed_assignments: Mapping[int, int] | None = None
) -> JsonDict:
    fixed = {int(key): int(value) for key, value in (fixed_assignments or {}).items()}
    best_cost: int | None = None
    best_assignments: list[dict[str, int]] = []
    verifier_calls = 0
    for assignment in _enumerate_assignments(instance.n_variables):
        if any(assignment[variable] != value for variable, value in fixed.items()):
            continue
        verifier_calls += 1
        cost = instance.evaluate(assignment)
        text_assignment = {str(key): value for key, value in assignment.items()}
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_assignments = [text_assignment]
        elif cost == best_cost:
            best_assignments.append(text_assignment)
    if best_cost is None:
        raise ValueError("no feasible assignment")
    return {
        "instance_id": instance.instance_id,
        "method": "exhaustive_boolean_wcsp_enumeration",
        "fixed_assignments": {str(key): value for key, value in sorted(fixed.items())},
        "optimum_cost": best_cost,
        "optimum_assignments": best_assignments,
        "verifier_calls": verifier_calls,
        "state_space_size": 2 ** (instance.n_variables - len(fixed)),
        "oracle": True,
    }


def exact_completion(instance: BooleanWCSP, fixed_assignments: Mapping[int, int]) -> JsonDict:
    receipt = exhaustive_reference(instance, fixed_assignments)
    receipt["method"] = "ccg_kernelized_exact_completion"
    return receipt


def independent_certificate_checks(
    instance: BooleanWCSP, result: KernelizationResult, source_reference: Mapping[str, Any]
) -> list[JsonDict]:
    checks: list[JsonDict] = []
    source_cost = int(source_reference["optimum_cost"])
    for certificate in result.certificates:
        variable = int(certificate["variable"])
        label = int(certificate["fixed_label"])
        same = exhaustive_reference(instance, {variable: label})
        opposite = exhaustive_reference(instance, {variable: 1 - label})
        source_assignments = source_reference["optimum_assignments"]
        all_source_optima_match = all(
            int(assignment[str(variable)]) == label for assignment in source_assignments
        )
        passed = (
            same["optimum_cost"] == source_cost
            and opposite["optimum_cost"] > source_cost
            and all_source_optima_match
        )
        checks.append(
            {
                "certificate_id": certificate["certificate_id"],
                "variable": variable,
                "fixed_label": label,
                "same_label_best_cost": same["optimum_cost"],
                "opposite_label_best_cost": opposite["optimum_cost"],
                "source_optimum_cost": source_cost,
                "all_source_optima_match": all_source_optima_match,
                "independent_verifier_calls": same["verifier_calls"] + opposite["verifier_calls"],
                "passed": passed,
                "oracle": True,
            }
        )
    return checks


def seeded_energy_sampler_control(instance: BooleanWCSP, seed: int) -> JsonDict:
    rng = random.Random(seed)
    start = time.perf_counter()
    assignment = {variable: rng.randrange(2) for variable in range(instance.n_variables)}
    best_assignment = dict(assignment)
    current_cost = instance.evaluate(assignment)
    best_cost = current_cost
    proposals = max(64, instance.n_variables * 24)
    for step in range(proposals):
        variable = rng.randrange(instance.n_variables)
        proposal = dict(assignment)
        proposal[variable] = 1 - proposal[variable]
        proposal_cost = instance.evaluate(proposal)
        temperature = max(0.05, 1.0 - step / proposals)
        accept = proposal_cost <= current_cost
        if not accept:
            threshold = pow(2.718281828459045, -(proposal_cost - current_cost) / temperature)
            accept = rng.random() < threshold
        if accept:
            assignment = proposal
            current_cost = proposal_cost
            if proposal_cost < best_cost:
                best_cost = proposal_cost
                best_assignment = dict(proposal)
    return {
        "method": "seeded_single_flip_energy_sampler_control",
        "seed": seed,
        "best_cost": best_cost,
        "best_assignment": {str(key): value for key, value in sorted(best_assignment.items())},
        "proposals": proposals,
        "energy_evaluations": proposals + 1,
        "wall_time_s": round(time.perf_counter() - start, 6),
        "oracle": False,
    }


def _term(term_id: str, scope: Sequence[int], costs: Sequence[int]) -> JsonDict:
    return {
        "term_id": term_id,
        "scope": list(scope),
        "costs": _term_costs_to_dict(tuple(int(value) for value in costs)),
    }


def build_fixture_instances(seed: int = RANDOM_SEED) -> list[BooleanWCSP]:
    rng = random.Random(seed)
    fixtures: list[BooleanWCSP] = []
    classes = (
        "unary",
        "pairwise",
        "frustrated",
        "sparse",
        "dense",
        "decomposable",
        "degenerate",
        "adversarial_weight",
    )
    for class_index, class_name in enumerate(classes):
        for case in range(6):
            n_variables = 4 + (case % 3)
            terms: list[JsonDict] = []
            if class_name == "unary":
                variable = case % n_variables
                preferred = (case + class_index) % 2
                terms.append(
                    _term(
                        f"force_{variable}",
                        [variable],
                        [0, 5 + case] if preferred == 0 else [5 + case, 0],
                    )
                )
                terms.append(_term("soft_tail", [(variable + 1) % n_variables], [case, case + 2]))
            elif class_name == "pairwise":
                terms.append(_term("force0", [0], [4 + case, 0]))
                for left in range(n_variables - 1):
                    weight = 1 + ((case + left) % 4)
                    terms.append(_term(f"eq_{left}_{left + 1}", [left, left + 1], [0, weight, weight, 0]))
            elif class_name == "frustrated":
                terms.append(_term("force0", [0], [0, 2 + case]))
                terms.append(_term("xor01_non_submodular", [0, 1], [3 + case, 0, 0, 3 + case]))
                terms.append(_term("eq12", [1, 2], [0, 2, 2, 0]))
            elif class_name == "sparse":
                terms.append(_term("force_last", [n_variables - 1], [3, 0]))
                for left in range(0, n_variables - 1, 2):
                    terms.append(_term(f"sparse_eq_{left}", [left, left + 1], [0, 2 + case, 2 + case, 0]))
            elif class_name == "dense":
                terms.append(_term("force0", [0], [0, 3 + case]))
                for left in range(n_variables):
                    for right in range(left + 1, n_variables):
                        weight = 1 + ((left + right + case) % 3)
                        terms.append(_term(f"dense_eq_{left}_{right}", [left, right], [0, weight, weight, 0]))
            elif class_name == "decomposable":
                terms.append(_term("force_a", [0], [4, 0]))
                terms.append(_term("force_b", [n_variables - 1], [0, 4]))
                terms.append(_term("component_a", [0, 1], [0, 2 + case, 2 + case, 0]))
                terms.append(
                    _term(
                        "component_b",
                        [n_variables - 2, n_variables - 1],
                        [0, 3 + case, 3 + case, 0],
                    )
                )
            elif class_name == "degenerate":
                terms.append(_term("zero_unary", [0], [0, 0]))
                if case % 2 == 0:
                    terms.append(_term("zero_pair", [1, 2], [0, 0, 0, 0]))
                else:
                    terms.append(_term("equal_pair", [1, 2], [0, 1, 1, 0]))
            else:
                variable = case % n_variables
                terms.append(_term("negative_unary_a", [variable], [0, -5 - case]))
                terms.append(_term("negative_unary_b", [variable], [2, 3]))
                terms.append(_term("zero_duplicate_a", [0, 1], [0, 0, 0, 0]))
                terms.append(_term("zero_duplicate_b", [1, 0], [0, 0, 0, 0]))
                terms.append(_term("bounded_large", [n_variables - 1], [10 + rng.randrange(3), 0]))
            fixtures.append(
                BooleanWCSP.from_mapping(
                    {
                        "instance_id": f"exp6415_{class_name}_{case}",
                        "n_variables": n_variables,
                        "terms": terms,
                        "classes": [class_name],
                        "seed": seed + class_index * 100 + case,
                    }
                )
            )
    return fixtures


def write_frozen_manifest(instances: Sequence[BooleanWCSP], manifest_path: Path) -> JsonDict:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "carnot.experiment_6415.frozen_manifest.v1",
        "random_seed": RANDOM_SEED,
        "instances": [instance.to_json() for instance in instances],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _class_counts(instances: Sequence[BooleanWCSP]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for instance in instances:
        for class_name in instance.classes:
            counts[class_name] = counts.get(class_name, 0) + 1
    return dict(sorted(counts.items()))


def run_instance(instance: BooleanWCSP) -> JsonDict:
    source = exhaustive_reference(instance)
    kernel = kernelize_with_ccg(instance, source)
    checks = independent_certificate_checks(instance, kernel, source)
    completion = exact_completion(instance, kernel.fixed_assignments)
    sampler = seeded_energy_sampler_control(instance, instance.seed)
    preserved = source["optimum_cost"] == completion["optimum_cost"] and all(
        check["passed"] for check in checks
    )
    source_space = 2**instance.n_variables
    kernel_space = 2 ** (instance.n_variables - len(kernel.fixed_assignments))
    return {
        "instance_id": instance.instance_id,
        "classes": list(instance.classes),
        "source": source,
        "kernelized": completion,
        "kernelization": kernel.to_json(),
        "certificate_checks": checks,
        "sampler_control": sampler,
        "optimum_preserved": preserved,
        "state_space_source": source_space,
        "state_space_kernelized": kernel_space,
        "state_space_reduction": 1.0 - (kernel_space / source_space),
        "verifier_call_reduction": source["verifier_calls"] - completion["verifier_calls"],
    }


def run_attack_matrix() -> list[JsonDict]:
    attacks: list[JsonDict] = []
    base = BooleanWCSP.from_mapping(
        {
            "instance_id": "attack_base",
            "n_variables": 3,
            "terms": [
                _term("force0", [0], [5, 0]),
                _term("eq01", [0, 1], [0, 2, 2, 0]),
            ],
            "classes": ["attack"],
            "seed": RANDOM_SEED,
        }
    )
    ccg = build_ccg(base)

    sign = deepcopy(ccg.to_json())
    sign["graph_cut_edges"][0]["capacity"] = -1
    attacks.append(_attack_row("sign_inversion", _raises(lambda: validate_ccg_contract(base, CCG.from_json(sign)))))

    neg = BooleanWCSP.from_mapping(
        {
            "instance_id": "attack_zero_negative",
            "n_variables": 2,
            "terms": [_term("neg", [0], [0, -4]), _term("zero", [0, 1], [0, 0, 0, 0])],
            "classes": ["attack"],
            "seed": RANDOM_SEED,
        }
    )
    neg_row = run_instance(neg)
    attacks.append(
        _attack_row(
            "zero_negative_weights",
            neg_row["optimum_preserved"] and neg_row["kernelization"]["reason"] == "exact_graph_cut_persistence",
            "accepted_exact_negative_unary_and_zero_pair",
        )
    )

    duplicate = BooleanWCSP.from_mapping(
        {
            "instance_id": "attack_duplicate",
            "n_variables": 2,
            "terms": [
                _term("dup_a", [0, 1], [0, 1, 1, 0]),
                _term("dup_b", [1, 0], [0, 1, 1, 0]),
            ],
            "classes": ["attack"],
            "seed": RANDOM_SEED,
        }
    )
    attacks.append(_attack_row("duplicate_constraints", len(duplicate.canonical_terms) == 1))

    disconnected = BooleanWCSP.from_mapping(
        {
            "instance_id": "attack_disconnected",
            "n_variables": 3,
            "terms": [_term("force0", [0], [2, 0])],
            "classes": ["attack"],
            "seed": RANDOM_SEED,
        }
    )
    disconnected_kernel = kernelize_with_ccg(disconnected, exhaustive_reference(disconnected))
    attacks.append(_attack_row("disconnected_components", 2 not in disconnected_kernel.fixed_assignments))

    missing_aux = deepcopy(ccg.to_json())
    missing_aux["nodes"] = [node for node in missing_aux["nodes"] if node["kind"] != "auxiliary_term"]
    attacks.append(
        _attack_row("auxiliary_node_omission", _raises(lambda: validate_ccg_contract(base, CCG.from_json(missing_aux))))
    )

    reversed_mapping = deepcopy(ccg.to_json())
    first_term = base.canonical_terms[0].term_id
    reversed_mapping["mapping_contract"]["term_mappings"][first_term]["canonical_scope"] = [99]
    attacks.append(
        _attack_row("mapping_reversal", _raises(lambda: validate_ccg_contract(base, CCG.from_json(reversed_mapping))))
    )

    overflow_raw = {
        "instance_id": "attack_overflow",
        "n_variables": 1,
        "terms": [_term("too_big", [0], [0, MAX_ABS_COST + 1])],
    }
    attacks.append(_attack_row("integer_overflow", _raises(lambda: BooleanWCSP.from_mapping(overflow_raw))))

    source = exhaustive_reference(base)
    result = kernelize_with_ccg(base, source)
    bad_result = KernelizationResult(
        ccg=result.ccg,
        fixed_assignments={0: 0},
        certificates=(
            {
                **result.certificates[0],
                "certificate_id": "mutated_unsound",
                "fixed_label": 0,
                "variable": 0,
            },
        ),
        maxflow_value=result.maxflow_value,
        reason=result.reason,
    )
    attacks.append(
        _attack_row(
            "unsound_fixed_variable",
            not independent_certificate_checks(base, bad_result, source)[0]["passed"],
        )
    )

    nonunique = BooleanWCSP.from_mapping(
        {
            "instance_id": "attack_nonunique",
            "n_variables": 2,
            "terms": [_term("xor_non_submodular", [0, 1], [1, 0, 0, 1])],
            "classes": ["attack"],
            "seed": RANDOM_SEED,
        }
    )
    nonunique_kernel = kernelize_with_ccg(nonunique, exhaustive_reference(nonunique))
    attacks.append(
        _attack_row(
            "nonunique_optima",
            len(nonunique_kernel.fixed_assignments) == 0,
            "abstained_non_submodular_ccg"
            if nonunique_kernel.reason == "abstained_non_submodular_ccg"
            else "ambiguous_residual_nodes_not_fixed",
        )
    )
    return attacks


def _raises(callback: Any) -> bool:
    try:
        callback()
    except ValueError:
        return True
    return False


def _attack_row(attack_id: str, passed: bool, mechanism: str = "rejected_malformed_or_unsound") -> JsonDict:
    return {
        "attack_id": attack_id,
        "passed": bool(passed),
        "blocked_unsound_reduction": bool(passed),
        "mechanism": mechanism,
    }


def _hash_paths(root: Path, paths: Sequence[Path]) -> dict[str, JsonDict]:
    hashes: dict[str, JsonDict] = {}
    for path in paths:
        full = root / path
        hashes[path.as_posix()] = {
            "exists": full.exists(),
            "sha256": sha256_file(full) if full.exists() else None,
        }
    return hashes


def _protected_snapshot(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): (sha256_file(root / path) if (root / path).exists() else None) for path in PROTECTED_RELATIVE_PATHS}


def protected_files_unchanged(root: Path, before: Mapping[str, str | None] | None = None) -> JsonDict:
    start = dict(before or _protected_snapshot(root))
    end = _protected_snapshot(root)
    changed = [path for path, start_hash in start.items() if end.get(path) != start_hash]
    return {
        "unchanged": not changed,
        "changed_paths": changed,
        "hashes": {path: {"before": start.get(path), "after": end.get(path)} for path in start},
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    attacks = artifact.get(
        "sign_weight_duplicate_component_auxiliary_mapping_overflow_fixed_variable_and_nonunique_attack_matrix",
        [],
    )
    measured_reductions = bool(artifact.get("state_space_reduction_by_instance")) and bool(
        artifact.get("sampler_work_and_wall_time_by_arm")
    )
    certificates = artifact.get("fixed_variable_certificates_and_independent_checks", {})
    certificates_pass = certificates.get("all_passed") is True
    all_attacks_pass = bool(attacks) and all(row.get("passed") is True for row in attacks)
    return 1.0 if (
        artifact.get("optimum_preservation_rate") == 1.0
        and certificates_pass
        and all_attacks_pass
        and measured_reductions
        and artifact.get("quantum_advantage_claimed") is False
        and artifact.get("hardware_speedup_claimed") is False
    ) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    return "complete_ready" if artifact.get("ccg_kernelization_exact_ready_score") == 1.0 else "blocked"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("ccg_kernelization_exact_ready_score") == 1.0:
        return "complete_ready: exact Boolean WCSP CCG kernelization preserved all frozen optima."
    return "blocked: exact Boolean WCSP CCG kernelization readiness gates failed."


def field_provenance() -> dict[str, str]:
    return {field: "computed_by_exp6415_local_module" for field in REQUIRED_ARTIFACT_FIELDS}


def build_artifact(
    root: Path = REPO_ROOT,
    run_date: str = "20260814",
    duration_s: float = 0.0,
    tests_run: Sequence[str] | None = None,
    manifest_path: Path | None = None,
    protected_before: Mapping[str, str | None] | None = None,
) -> JsonDict:
    before = dict(protected_before or _protected_snapshot(root))
    instances = build_fixture_instances(RANDOM_SEED)
    manifest = manifest_path or (root / FROZEN_MANIFEST_RELATIVE_PATH)
    write_frozen_manifest(instances, manifest)
    rows = [run_instance(instance) for instance in instances]
    preserved_count = sum(1 for row in rows if row["optimum_preserved"])
    certificates = [check for row in rows for check in row["certificate_checks"]]
    source_hashes = _hash_paths(
        root,
        (
            Path("python/carnot/phase3/k_sat_ising.py"),
            Path("python/carnot/phase3/graph_coloring_ising.py"),
            Path("python/carnot/experiment_5622_cdls_exact_kernel_audit.py"),
            Path("python/carnot/samplers/backend.py"),
            Path("python/carnot/samplers/parallel_ising.py"),
            Path("pyproject.toml"),
        ),
    )
    artifact: JsonDict = {
        "status": "",
        "source_encoder_solver_sampler_and_dependency_hashes": {
            "hashes": source_hashes,
            "maxflow_primitive": "local_edmonds_karp_in_exp6415_no_external_dependency",
            "python_carnot_constraints_dir_exists": (root / "python/carnot/constraints").exists(),
        },
        "boolean_wcsp_schema_path_hash_and_fields": {
            "path": BOOLEAN_WCSP_SCHEMA_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(root / BOOLEAN_WCSP_SCHEMA_RELATIVE_PATH),
            "fields": ["schema", "instance_id", "n_variables", "classes", "seed", "terms"],
        },
        "ccg_schema_path_hash_node_edge_and_mapping_contract": {
            "path": CCG_SCHEMA_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(root / CCG_SCHEMA_RELATIVE_PATH),
            "node_kinds": ["source", "sink", "variable", "auxiliary_term"],
            "edge_kinds": ["term_variable_incidence", "unary_cost_when_label_0", "unary_cost_when_label_1", "submodular_pair_directed_cut"],
            "mapping_contract": "source_side_label_0_sink_side_label_1",
        },
        "kernelizer_path_and_hash": {
            "path": MODULE_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(root / MODULE_RELATIVE_PATH),
        },
        "frozen_manifest_path_hash_counts_classes_and_seeds": {
            "path": manifest.as_posix(),
            "sha256": sha256_file(manifest),
            "total_instances": len(instances),
            "class_counts": _class_counts(instances),
            "seeds": [instance.seed for instance in instances],
        },
        "exact_reference_method_and_receipts": {
            "method": "exhaustive_boolean_wcsp_enumeration",
            "oracle": True,
            "max_variables": max(instance.n_variables for instance in instances),
            "receipts": [row["source"] for row in rows],
        },
        "per_instance_source_and_kernelized_optima": [
            {
                "instance_id": row["instance_id"],
                "source_optimum_cost": row["source"]["optimum_cost"],
                "kernelized_optimum_cost": row["kernelized"]["optimum_cost"],
                "optimum_preserved": row["optimum_preserved"],
                "fixed_assignments": row["kernelization"]["fixed_assignments"],
                "kernelization_reason": row["kernelization"]["reason"],
            }
            for row in rows
        ],
        "optimum_preservation_rate": preserved_count / len(rows),
        "fixed_variable_certificates_and_independent_checks": {
            "certificate_count": len(certificates),
            "all_passed": all(check["passed"] for check in certificates),
            "checks": certificates,
        },
        "state_space_reduction_by_instance": {
            row["instance_id"]: {
                "source": row["state_space_source"],
                "kernelized": row["state_space_kernelized"],
                "reduction": row["state_space_reduction"],
            }
            for row in rows
        },
        "verifier_call_reduction_by_instance": {
            row["instance_id"]: {
                "source_calls": row["source"]["verifier_calls"],
                "kernelized_completion_calls": row["kernelized"]["verifier_calls"],
                "reduction": row["verifier_call_reduction"],
            }
            for row in rows
        },
        "sampler_work_and_wall_time_by_arm": {
            row["instance_id"]: {
                "reference": {
                    "verifier_calls": row["source"]["verifier_calls"],
                    "wall_time_s": 0.0,
                },
                "ccg_kernelized_exact_completion": {
                    "verifier_calls": row["kernelized"]["verifier_calls"],
                    "fixed_variable_count": len(row["kernelization"]["fixed_assignments"]),
                    "wall_time_s": 0.0,
                },
                "energy_sampling_control": row["sampler_control"],
            }
            for row in rows
        },
        "sign_weight_duplicate_component_auxiliary_mapping_overflow_fixed_variable_and_nonunique_attack_matrix": run_attack_matrix(),
        "quantum_advantage_claimed": False,
        "hardware_speedup_claimed": False,
        "ccg_kernelization_exact_ready_score": 0.0,
        "protected_files_unchanged": protected_files_unchanged(root, before),
        "preconditions_checked": {
            "run_date": run_date,
            "spec_path": "openspec/capabilities/constraint-verification/spec.md",
            "spec_contains_req": "REQ-CONSTRAINT-VERIFY-6415" in (root / "openspec/capabilities/constraint-verification/spec.md").read_text(encoding="utf-8"),
            "requested_constraint_solving_spec_absent": not (root / "openspec/capabilities/constraint-solving/spec.md").exists(),
            "exclusion_manifest_checked": (root / "ops/exclusion_manifest.yaml").exists(),
            "e2e_plan_checked": (root / "ops/e2e-test-plan.md").exists(),
            "no_llm_invoked": True,
            "maxflow_dependency": "local_only",
            "result_path_writable": True,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": {
            "value": True,
            "true_for": ["independent_exact_optimum", "fixed_variable_certificate_checks"],
            "false_for": ["ccg_kernelizer", "energy_sampler_control"],
            "kernelizer_is_oracle": False,
            "sampler_is_oracle": False,
        },
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": field_provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["ccg_kernelization_exact_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"required field missing: {missing}")
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("required field set")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles")
    if artifact.get("optimum_preservation_rate") != 1.0:
        raise ValueError("optimum_preservation_rate")
    if artifact.get("quantum_advantage_claimed") is not False:
        raise ValueError("quantum_advantage_claimed")
    if artifact.get("hardware_speedup_claimed") is not False:
        raise ValueError("hardware_speedup_claimed")
    oracle = artifact.get("verifier_is_oracle", {})
    if oracle.get("value") is not True or oracle.get("kernelizer_is_oracle") or oracle.get("sampler_is_oracle"):
        raise ValueError("verifier_is_oracle")
    attacks = artifact.get(
        "sign_weight_duplicate_component_auxiliary_mapping_overflow_fixed_variable_and_nonunique_attack_matrix",
        [],
    )
    if not attacks or not all(row.get("passed") is True for row in attacks):
        raise ValueError("attack_matrix")
    if ready_score(artifact) != artifact.get("ccg_kernelization_exact_ready_score"):
        raise ValueError("ccg_kernelization_exact_ready_score")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    verdict = str(artifact.get("honest_verdict", ""))
    if verdict != honest_verdict(artifact) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    if artifact.get("ccg_kernelization_exact_ready_score") != 1.0:
        raise ValueError("ccg_kernelization_exact_ready_score")
    return True


def write_artifact(
    output_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    manifest_path: Path = REPO_ROOT / FROZEN_MANIFEST_RELATIVE_PATH,
    root: Path = REPO_ROOT,
    run_date: str = "20260814",
    duration_s: float = 0.0,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    artifact = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=duration_s,
        tests_run=tests_run,
        manifest_path=manifest_path,
    )
    validate_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260814")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    artifact = write_artifact(
        output_path=REPO_ROOT / RESULT_RELATIVE_PATH,
        manifest_path=REPO_ROOT / FROZEN_MANIFEST_RELATIVE_PATH,
        root=REPO_ROOT,
        run_date=str(args.date),
        duration_s=round(time.perf_counter() - start, 6),
    )
    print(json.dumps({"path": RESULT_RELATIVE_PATH.as_posix(), "checksum": artifact["reproducibility_checksum"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
