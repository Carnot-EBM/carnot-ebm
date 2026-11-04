"""REQ-REPORT-8803: seal future worlds while qualifying only explicit truth controls."""

from copy import deepcopy
import sys
from types import CodeType
from typing import Any

from carnot.reporting import evidence_intervention_methods_8791 as old
from carnot.reporting import evidence_intervention_corpus_8791 as c
from carnot.reporting import evidence_set_corpus_8768 as oracle
from carnot.reporting import evidence_set_methods_8768 as original
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v746_retained_evidence import reference, load

Json = dict[str, Any]
__all__ = ["old", "c", "oracle", "original"]
ROOT = old.ROOT
NAME, TASK = "experiment_8803_v754_repaired_evidence_methods", "exp8803-repaired-evidence-methods"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    f"python/carnot/reporting/{p}_8803.py"
    for p in ["repaired_evidence", "repaired_evidence_runner"]
] + [CLI]
TESTS = ["tests/python/test_repaired_evidence_8803.py"]
SEED, BUDGET = 7548803, 200000
MODEL_SPECS: list[Json] = []
LABELS = ("support", "refute", "insufficient")
CONDITIONS = (
    "intact_evidence",
    "irrelevant_padding",
    "reordered_evidence",
    "missing_necessary_hop",
    "conflicting_fact",
    "role_binding_swap",
)
GROUPS = ("unanimous_correct", "unanimous_wrong", "disagreement")
HISTORY = ROOT / "results/experiment_8791_v753_evidence_intervention_methods.json"
HISTORY_PIN = "sha256:7fabfb973c9ce2b0c9ea38c285da0fe08bdc3ba5d1713bb40cf6b03c58781533"
ARCHIVE = "openspec/change-proposals/research-roadmap-v753-preserved-20261104.md"
ARCHIVE_PIN = "sha256:15b63fe517179897398afe775d4c136e0b453fc3b1b9bb57137dd2ada3f78d8d"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so waiting parents can distinguish progress from a stall."""
    print(f"[exp8803] phase={phase} completed={completed} pending={pending}", flush=True)


def variants(edges: Any) -> list[Json]:
    """Balance three truth classes without treating nuisance siblings as independent worlds."""
    nodes = ["?x", "a", "b", "c", "d"]
    query = [[nodes[a], f"r{i}", nodes[b]] for i, (a, b) in enumerate(edges)]
    base = [["w" if t == "?x" else t for t in q] + [True] for q in query]
    hop = next(i for i, edge in enumerate(edges) if 0 in edge)
    position = 0 if edges[hop][0] == 0 else 2
    ground = [i for i, (a, b) in enumerate(edges) if a and b]
    rows = []
    names = [
        *CONDITIONS,
        "alternative_proof",
        "absence_control",
        *[f"refute_control_{i}" for i in range(4)],
    ]
    for name in names:
        facts = deepcopy(base)
        expected = "support"
        if name in {"missing_necessary_hop", "absence_control"}:
            facts.pop(hop if name == "missing_necessary_hop" else ground[0])
            expected = "insufficient"
        elif name == "conflicting_fact":
            facts.append([*base[hop][:3], False])
            expected = "insufficient"
        elif name == "role_binding_swap":
            facts[hop][position] = "z"
            expected = "insufficient"
        elif name == "irrelevant_padding":
            facts += [["c", "noise", "d", True]] * 2
        elif name == "alternative_proof":
            facts[hop][3] = False
            facts += [["z" if t == "w" else t for t in q[:3]] + [True] for q in base]
        elif name.startswith("refute_control_"):
            facts[ground[int(name[-1]) % len(ground)]][3] = False
            expected = "refute"
        if name == "reordered_evidence":
            facts.reverse()
        rows.append(dict(condition=name, expected=expected, input=dict(facts=facts, query=query)))
    return rows


def views(rows: list[Json], world: str) -> list[Json]:
    """Protocol lengths and evidence order vary independently of truth-bearing records."""
    result = []
    for i, row in enumerate(rows):
        for order in ("forward", "reverse"):
            for target in c.BINS:
                payload = c.pad(row["input"], target)
                if order == "reverse":
                    payload = {k: list(reversed(v)) for k, v in payload.items()}
                result.append(
                    dict(
                        row,
                        input=payload,
                        world_id=world,
                        unit_id=f"{world}/{i}/{order}/{target}",
                        ordering=order,
                        token_bin=target,
                    )
                )
    return result


def freeze() -> Json:
    """Canonical rejection excludes even renamed historical worlds before future labels open."""
    historical = set(c.roster()["historical_structure_hashes"])
    historical.update(w["structure"] for w in c.roster()["worlds"])
    worlds: list[Json] = []
    collisions = 0
    counts = dict(intervention=120, consensus_dev=60, consensus_test=120, canary=12)
    for edge_count in (11, 12, 13):
        available = []
        for edges in oracle.graph_roster(edge_count, 144):
            structure = canonical_hash(edges)
            if structure in historical:
                collisions += 1
            else:
                available.append((edges, structure))
        offset = 0
        for split, count in counts.items():
            for edges, structure in available[offset : offset + count // 3]:
                worlds.append(
                    dict(
                        world_id=f"v754/{split}/{edge_count}/{structure.split(':')[-1][:16]}",
                        split=split,
                        edges=edges,
                        structure=structure,
                        edge_count=edge_count,
                    )
                )
            offset += count // 3
    intervention, consensus, balance = [], [], []
    for i, world in enumerate(worlds):
        rows = variants(world["edges"])
        if world["split"] == "intervention":
            group = views(rows, world["world_id"])
            intervention += group
            for order in ("forward", "reverse"):
                for target in c.BINS:
                    selected = [
                        r for r in group if r["ordering"] == order and r["token_bin"] == target
                    ]
                    label_counts = {
                        label: sum(r["expected"] == label for r in selected) for label in LABELS
                    }
                    balance.append(
                        dict(
                            world_id=world["world_id"],
                            ordering=order,
                            token_bin=target,
                            label_counts=label_counts,
                            passed=set(label_counts.values()) == {4}
                            and all(c.tokens(r["input"]) == target for r in selected),
                        )
                    )
        elif world["split"] != "canary":
            seeds = [rows[0], rows[7], rows[8]]
            candidates = []
            for row in seeds:
                label = row["expected"]
                wrong = LABELS[(LABELS.index(label) + 1) % 3]
                for name in GROUPS:
                    answers = (
                        [label] * 3
                        if name == "unanimous_correct"
                        else [wrong] * 3
                        if name == "unanimous_wrong"
                        else list(LABELS)
                    )
                    candidates.append(
                        dict(
                            row,
                            group=name,
                            candidates=answers,
                            sufficient=label != "insufficient",
                            split=world["split"],
                        )
                    )
            group = views(candidates, world["world_id"])
            consensus += group
            for order in ("forward", "reverse"):
                for target in c.BINS:
                    selected = [
                        r for r in group if r["ordering"] == order and r["token_bin"] == target
                    ]
                    label_counts = {
                        label: sum(r["expected"] == label for r in selected) for label in LABELS
                    }
                    balance.append(
                        dict(
                            world_id=world["world_id"],
                            ordering=order,
                            token_bin=target,
                            label_counts=label_counts,
                            passed=set(label_counts.values()) == {3}
                            and all(c.tokens(r["input"]) == target for r in selected),
                        )
                    )
        if (i + 1) % 20 == 0:
            progress("seal_worlds", i + 1, len(worlds) - i - 1)
    manifest = dict(
        worlds=worlds,
        counts=counts,
        collision_rejections=collisions,
        historical_structures=sorted(historical),
        disjoint=len(worlds) == len({w["structure"] for w in worlds}) == 312,
        measured_count=0,
        canonicalization="directed graph; variable vertex distinguished; constants permuted",
        labels_opened_for_evaluation=False,
    )
    return dict(
        split_manifest=manifest,
        intervention=intervention,
        consensus=consensus,
        length_balance_rows=balance,
    )


def controls() -> list[Json]:
    """Small explicit witnesses check the reference independently of future study labels."""
    rows = variants(((0, 1), (0, 2), (1, 2), (2, 3)))
    return rows + [
        dict(condition="hand_inspected_counterexample", input=p, expected=label)
        for p, label in c.counterexamples()
    ]


def repair_controls(private: Any) -> list[Json]:
    """Exercise the repaired original authentication boundary without replacing its validator."""
    receipts = []
    cases: list[tuple[str, Any, str]] = [
        ("null", None, "rejected_producer"),
        ("scalar", 7, "rejected_producer"),
        ("pairs", [["evidence_methods_ready_score", 1]], "rejected_producer"),
        ("empty", {}, "missing_field"),
        ("zero", dict(evidence_methods_ready_score=0), "observed_zero"),
        ("byte_drift", dict(evidence_methods_ready_score=1), "rejected_producer"),
    ]
    for name, value, expected in cases:
        path = private / (name + ".json")
        atomic_json(path, value)
        result = old.authenticate(path, old.UPSTREAM_ROWS)
        receipts.append(
            dict(
                control=name,
                expected=expected,
                observed=result["diagnostics"][0]["check"],
                passed=not result["passed"] and result["diagnostics"][0]["check"] == expected,
                source=reference(path),
            )
        )
    return receipts


class BudgetExceeded(Exception):
    """An unfinished proof must abstain rather than exceed the frozen work allowance."""


class PrimitiveCounter:
    """A directly testable callback keeps untraceable interpreter hooks accountable."""

    def __init__(self, limit: int) -> None:
        self.count, self.limit = 0, limit
        self.solver_directory = str(ROOT / "python/carnot/reporting") + "/"

    def __call__(self, code: CodeType, line: int) -> None:
        """A separate monitoring slot leaves Coverage.py's real trace intact."""
        if code.co_filename == __file__ or not code.co_filename.startswith(self.solver_directory):
            return
        if self.count == self.limit:
            sys.monitoring.set_events(4, 0)
            raise BudgetExceeded
        self.count += 1


def bounded(arm: str, payload: Json, limit: int = BUDGET) -> Json:
    """Count the same Python line dispatch primitive inside each unchanged solver."""
    original.parse(payload)
    counter = PrimitiveCounter(limit)

    sys.monitoring.use_tool_id(4, "exp8803-budget")
    sys.monitoring.register_callback(4, sys.monitoring.events.LINE, counter)
    exhausted = False
    try:
        sys.monitoring.set_events(4, sys.monitoring.events.LINE)
        decision, calls = old.evaluate(arm, payload)
    except BudgetExceeded:
        decision, calls, exhausted = "insufficient", None, True
    finally:
        sys.monitoring.set_events(4, 0)
        sys.monitoring.free_tool_id(4)
    return dict(
        arm=arm,
        decision=decision,
        evaluator_invocations=calls,
        primitive_operations=counter.count,
        primitive_limit=limit,
        budget_exhausted=exhausted,
    )


def measure(payloads: list[Json]) -> list[Json]:
    """Only records and query terms reach solvers; unlimited truth is a separate arm."""
    rows = []
    for i, payload in enumerate(payloads):
        original.parse(payload)
        for arm in old.ARMS:
            rows.append(
                dict(bounded(arm, payload), input_index=i, input_sha256=canonical_hash(payload))
            )
        rows.append(
            dict(
                arm="unlimited_exact_reference",
                decision=old.strong(payload),
                input_index=i,
                input_sha256=canonical_hash(payload),
                primitive_limit=None,
            )
        )
        if (i + 1) % 10 == 0:
            progress("truth_controls", i + 1, len(payloads) - i - 1)
    return rows
