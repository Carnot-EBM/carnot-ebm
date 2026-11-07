"""REQ-REPORT-8818: measure decisive evidence under equal explicit work ceilings."""

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import time
from typing import Any

from carnot.reporting import evidence_intervention_methods_8791 as old
from carnot.reporting import evidence_operation_meter_8818 as meter
from carnot.reporting import v749_verifier_protocol as numerical

__all__ = ["old", "meter"]
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.v746_retained_evidence import reference

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8818_v755_evidence_decisive_coverage", "exp8818-evidence-decisive-coverage"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    f"python/carnot/reporting/{s}_8818.py"
    for s in ["evidence_operation_meter", "evidence_decisive", "evidence_decisive_runner"]
] + [CLI]
TESTS = ["tests/python/test_evidence_decisive_coverage_8818.py"]
ARMS, SEED, BUDGET = old.ARMS, 7558818, 200000
UPSTREAM = ROOT / "results/experiment_8816_v755_selective_evidence_methods.json"
PIN = "sha256:3a474cb1eab956509047d686bf11c0998df99e536f2634dc1e2f874859ae7898"
CONDITIONS = (
    "intact",
    "irrelevant_padding",
    "reordered_facts",
    "missing_hop",
    "conflicting_fact",
    "binding_swap",
)
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so a supervisor can distinguish work from a stalled child."""
    print(f"[exp8818] phase={phase} completed={completed} pending={pending}", flush=True)


def authenticate(path: Path) -> Json:
    """Pinned producer and bound sources prevent a bare readiness flag granting admission."""
    diagnostic: Json = dict(
        check="missing_file",
        upstream="exp8816-selective-evidence-methods",
        artifact_field="evidence_protocol_ready_score",
        expected=1,
        observed=None,
        path=str(path),
        sha256=None,
    )
    try:
        diagnostic.update(check="rejected_producer", sha256=sha256_file(path))
        value = json.loads(path.read_bytes())
        if not isinstance(value, dict):
            raise ValueError("producer_object")
        field = "evidence_protocol_ready_score"
        diagnostic.update(check="missing_field", observed=value.get(field))
        if field not in value:
            raise ValueError("missing_field")
        diagnostic["check"] = "observed_zero"
        if value[field] != 1:
            raise ValueError("readiness_zero")
        diagnostic.update(
            check="rejected_producer",
            artifact_field="sha256",
            expected=PIN,
            observed=diagnostic["sha256"],
        )
        if diagnostic["sha256"] != PIN or value["verdict_class"] != "circular_positive":
            raise ValueError("producer_identity")
        refs = (
            [reference(path)]
            + value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        )
        for ref in refs:
            diagnostic.update(path=ref["path"], expected=ref["sha256"])
            observed = sha256_file(Path(ref["path"]))
            diagnostic.update(observed=observed, sha256=observed)
            if observed != ref["sha256"]:
                raise ValueError("source_drift")
        return dict(passed=True, references=refs, diagnostics=[])
    except (OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, OSError):
            diagnostic["check"] = "missing_file"
        return dict(passed=False, references=[], diagnostics=[dict(diagnostic, error=str(error))])


def variants(edges: list[list[int]], variables: int, alternative: bool) -> list[Json]:
    """Alternative complete witnesses cannot be rejected because one old witness failed."""
    names = [f"?v{i}" if i < variables else f"n{i}" for i in range(8)]
    query = [[names[a], f"r{i}", names[b]] for i, (a, b) in enumerate(edges)]
    base = [[f"n{int(t[2:])}" if t.startswith("?") else t for t in q] + [True] for q in query]
    if alternative:
        base = [[*base[0][:3], False]] + [
            [f"a{int(t[2:])}" if t.startswith("?") else t for t in q] + [True] for q in query
        ]
    result = []
    for condition in CONDITIONS:
        facts = deepcopy(base)
        witness = 1 if alternative else 0
        if condition == "irrelevant_padding":
            facts += [["n6", "noise", "n7", True]] * 8
        if condition == "reordered_facts":
            facts.reverse()
        if condition == "missing_hop":
            facts.pop(witness)
        if condition == "conflicting_fact":
            facts.append([*facts[witness][:3], False])
        if condition == "binding_swap":
            facts[witness][0] = "n7"
        result.append(
            dict(
                condition=condition,
                valid_alternative=alternative,
                input=dict(facts=facts, query=query),
            )
        )
    return result


def cohort(count: int, split: str) -> Json:
    """Reserve calibration invariants before certifying distinct evaluation structures."""
    rng = random.Random(SEED + (0 if split == "calibration" else 1))
    worlds: list[Json] = []
    inputs: list[Json] = []
    seen: set[str] = (
        {w["structure"] for w in cohort(60, "calibration")["worlds"]}
        if split == "evaluation"
        else set()
    )
    universe = [(a, b) for a in range(8) for b in range(8) if a != b]
    ring = [(i, (i + 1) % 8) for i in range(8)]
    while len(worlds) < count:
        edges = sorted(ring + rng.sample([e for e in universe if e not in ring], rng.randint(2, 9)))
        colors = [
            canonical_hash([sum(a == i for a, b in edges), sum(b == i for a, b in edges)])
            for i in range(8)
        ]
        for _ in range(2):
            colors = [
                canonical_hash(
                    [
                        colors[i],
                        sorted(colors[b] for a, b in edges if a == i),
                        sorted(colors[a] for a, b in edges if b == i),
                    ]
                )
                for i in range(8)
            ]
        structure = canonical_hash(sorted(colors))
        if structure in seen:
            continue
        seen.add(structure)
        index = len(worlds)
        wid = f"v755/8818/{split}/{index:03d}"
        variables = 2 + index % 2
        world: Json = dict(
            world_id=wid,
            vertices=8,
            variables=variables,
            structure=structure,
            edges=[list(e) for e in edges],
        )
        worlds.append(world)
        for row in variants(world["edges"], variables, index % 3 == 0):
            inputs.append(dict(row, unit_id=wid + "/" + row["condition"], cluster_id=wid))
    return dict(worlds=worlds, inputs=inputs, split=split)


def label(payload: Json) -> str:
    """An independent scan evaluator checks signed truth without scheduling compared methods."""
    signs: dict[tuple[str, ...], set[bool]] = {}
    for *atom, sign in payload["facts"]:
        signs.setdefault(tuple(atom), set()).add(sign)
    if any(len(s) > 1 for s in signs.values()):
        return "insufficient"
    return {"true": "support", "false": "refute", "unknown": "insufficient"}[
        numerical.truth(payload["facts"], payload["query"])
    ]


def measure(inputs: list[Json]) -> list[Json]:
    """Only facts and query reach each arm; exact costs remain outside budgeted rows."""
    progress("before_benchmark", 0, len(inputs) * 4)
    rows = []
    last = time.monotonic()
    for index, unit in enumerate(inputs):
        for arm in (*ARMS, "unrestricted_exact_reference"):
            exact = arm == "unrestricted_exact_reference"
            rows.append(
                dict(
                    meter.run(
                        "joint_symbolic" if exact else arm, unit["input"], None if exact else BUDGET
                    ),
                    unit_id=unit["unit_id"],
                    arm=arm,
                    input_sha256=canonical_hash(unit["input"]),
                )
            )
        if time.monotonic() - last >= 30 or (index + 1) % 120 == 0:
            progress("benchmark", len(rows), (len(inputs) - index - 1) * 4)
            last = time.monotonic()
    progress("after_benchmark", len(rows), 0)
    return rows


def score(cohort: Json, decisions: list[Json]) -> list[Json]:
    """Open labels only after all attempts seal; censored proofs retain denominator one."""
    expected = {
        (u["unit_id"], arm)
        for u in cohort["inputs"]
        for arm in (*ARMS, "unrestricted_exact_reference")
    }
    if len(decisions) != len(expected) or {(r["unit_id"], r["arm"]) for r in decisions} != expected:
        raise ValueError("attempt_roster")
    progress("before_independent_labels", 0, len(cohort["inputs"]))
    labels = {}
    last = time.monotonic()
    for i, unit in enumerate(cohort["inputs"]):
        labels[unit["unit_id"]] = (unit, label(unit["input"]))
        if time.monotonic() - last >= 30:
            progress("independent_labels", i + 1, len(cohort["inputs"]) - i - 1)
            last = time.monotonic()
    rows = []
    for decision in decisions:
        unit, truth = labels[decision["unit_id"]]
        if decision["arm"] == "unrestricted_exact_reference" and decision["decision"] != truth:
            raise ValueError("independent_reference_disagreement")
        decisive = decision["decision"] != "insufficient"
        rows.append(
            dict(
                decision,
                cluster_id=unit["cluster_id"],
                seed=SEED,
                condition=unit["condition"],
                valid_alternative=unit["valid_alternative"],
                truth=truth,
                metric=int(decisive and decision["decision"] == truth),
                denominator=1,
                raw_numerator=int(decisive and decision["decision"] == truth),
                raw_denominator=1,
                status="censored" if decision["budget_exhausted"] else "completed",
                missing_reason="operation_budget_exhausted"
                if decision["budget_exhausted"]
                else None,
            )
        )
    progress("after_independent_labels", len(labels), 0)
    return rows


def metrics(rows: list[Json]) -> Json:
    """Useful decisions and errors share explicit denominators, including abstentions."""
    n = len(rows)
    unknown = [r for r in rows if r["truth"] == "insufficient"]
    unsupported = [r for r in rows if r["truth"] != "support"]
    decisive = [r for r in rows if r["decision"] != "insufficient"]
    alternative = [r for r in rows if r["valid_alternative"] and r["truth"] == "support"]
    return dict(
        denominator=n,
        correct_decisive_coverage=sum(
            r["decision"] == r["truth"] and r["decision"] != "insufficient" for r in rows
        )
        / n,
        semantic_false_accept_rate=sum(r["decision"] == "support" for r in unsupported)
        / len(unsupported)
        if unsupported
        else None,
        false_accept_denominator=len(unsupported),
        insufficient_denominator=len(unknown),
        insufficient_to_decision_rate=sum(r["decision"] != "insufficient" for r in unknown)
        / len(unknown)
        if unknown
        else None,
        wrong_decision_rate=sum(r["decision"] != r["truth"] for r in decisive) / len(decisive)
        if decisive
        else None,
        valid_alternative_denominator=len(alternative),
        valid_alternative_false_rejects=sum(r["decision"] == "refute" for r in alternative),
        valid_alternative_abstentions=sum(r["decision"] == "insufficient" for r in alternative),
        unknown_refutations=sum(r["decision"] == "refute" for r in unknown),
    )


def select(rows: list[Json]) -> Json:
    """Deterministic scores are zero or one, so all six frozen thresholds give identical actions."""
    if any("/calibration/" not in r["unit_id"] for r in rows):
        raise ValueError("calibration_only")
    menu: list[Json] = []
    for arm in ARMS:
        observed = metrics([r for r in rows if r["arm"] == arm])
        groups: dict[str, list[Json]] = defaultdict(list)
        for row in rows:
            if row["arm"] == arm:
                groups[row["cluster_id"]].append(row)
        summaries = []
        for group in groups.values():
            decisive = [r for r in group if r["decision"] != "insufficient"]
            summaries.append(
                (
                    len(group),
                    len(decisive),
                    sum(r["decision"] != r["truth"] for r in decisive),
                    sum(r["decision"] == r["truth"] for r in decisive),
                )
            )
        rng = random.Random(SEED)
        risks, coverages = [], []
        for _ in range(2000):
            sample = rng.choices(summaries, k=len(summaries))
            n, decisive_count, wrong, correct = [sum(v[i] for v in sample) for i in range(4)]
            risks.append(wrong / decisive_count if decisive_count else 1.0)
            coverages.append(correct / n)
        risk_upper, coverage_lower = sorted(risks)[1987], sorted(coverages)[12]
        menu.append(
            dict(
                arm=arm,
                metrics=observed,
                threshold=0.5,
                threshold_menu=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                risk_upper=risk_upper,
                coverage_lower=coverage_lower,
                eligible=risk_upper <= 0.05 and coverage_lower >= 0.30,
            )
        )
    safe = [r for r in menu if r["eligible"]]
    best = max(safe, key=lambda r: r["metrics"]["correct_decisive_coverage"]) if safe else None
    return dict(
        comparator=best["arm"] if best else None,
        menu=menu,
        calibration_rows_sha256=canonical_hash(rows),
        selection_before_evaluation=True,
    )


def reduce(rows: list[Json], comparator: str) -> Json:
    """Resample complete paired worlds rather than pretending nuisance siblings are independent."""
    groups: dict[tuple[str, str], list[Json]] = defaultdict(list)
    for row in rows:
        groups[row["cluster_id"], row["arm"]].append(row)
    worlds = sorted({r["cluster_id"] for r in rows})
    per_world, vectors = [], []
    for world in worlds:
        local = {}
        for arm in ARMS:
            arm_rows = groups[world, arm]
            values = metrics(arm_rows)
            lookup = {r["condition"]: r for r in arm_rows}
            values["nuisance_invariance"] = (
                sum(lookup[c]["decision"] == lookup["intact"]["decision"] for c in CONDITIONS[1:3])
                / 2
            )
            local[arm] = values
            per_world.append(dict(cluster_id=world, arm=arm, **values))
        target, base = local["evidence_set"], local[comparator]
        vectors.append(
            [
                target["correct_decisive_coverage"] - base["correct_decisive_coverage"],
                target["semantic_false_accept_rate"] - base["semantic_false_accept_rate"],
                target["nuisance_invariance"],
                target["insufficient_to_decision_rate"],
            ]
        )
    rng = random.Random(SEED)
    draws: list[list[float]] = [[], [], [], []]
    for _ in range(2000):
        sample = rng.choices(vectors, k=len(vectors))
        for i in range(4):
            draws[i].append(sum(v[i] for v in sample) / len(sample))
    intervals = {
        name: dict(
            estimate=sum(v[i] for v in vectors) / len(vectors),
            lower=sorted(draws[i])[12],
            upper=sorted(draws[i])[1987],
            confidence=0.9875,
            independent_worlds=len(worlds),
            draws=2000,
            cluster="whole world",
        )
        for i, name in enumerate(
            [
                "coverage_gain",
                "semantic_false_accept_difference",
                "nuisance_invariance",
                "insufficient_to_decision_rate",
            ]
        )
    }
    gates = dict(
        coverage_gain=intervals["coverage_gain"]["lower"] > 0.05,
        semantic_false_accept=intervals["semantic_false_accept_difference"]["upper"] <= 0.01,
        nuisance_invariance=intervals["nuisance_invariance"]["lower"] >= 0.99,
        insufficient_truth_escalated=intervals["insufficient_to_decision_rate"]["upper"] <= 0.01,
    )
    budgeted = [r for r in rows if r["arm"] in ARMS]
    return dict(
        per_world_results=per_world,
        per_defect_rows=[
            dict(
                arm=arm,
                condition=c,
                **metrics([r for r in budgeted if r["arm"] == arm and r["condition"] == c]),
            )
            for arm in ARMS
            for c in CONDITIONS
        ],
        operation_budget_rows=[
            {
                k: r[k]
                for k in [
                    "unit_id",
                    "arm",
                    "primitive_limit",
                    "primitive_operations",
                    "operation_counts",
                    "budget_exhausted",
                    "wall_time_s",
                    "cpu_time_s",
                ]
            }
            for r in budgeted
        ],
        whole_world_intervals=intervals,
        acceptance_gates=gates,
        exact_reference_rows=[r for r in rows if r["arm"] not in ARMS],
        exhausted_count=sum(r["budget_exhausted"] for r in budgeted),
    )


def controls() -> list[Json]:
    """Hand-inspected controls prevent a circular reference from hiding changed semantics."""
    cases: list[Json] = [
        dict(facts=[["a", "r", "b", True]], query=[["?x", "r", "b"]]),
        dict(facts=[["a", "r", "b", False]], query=[["a", "r", "b"]]),
        dict(facts=[], query=[["a", "r", "b"]]),
    ]
    return [
        dict(
            control=f"unchanged_{arm}_{i}",
            passed=meter.run(arm, payload, None)["decision"] == old.evaluate(arm, payload)[0],
        )
        for i, payload in enumerate(cases)
        for arm in ARMS
    ]
