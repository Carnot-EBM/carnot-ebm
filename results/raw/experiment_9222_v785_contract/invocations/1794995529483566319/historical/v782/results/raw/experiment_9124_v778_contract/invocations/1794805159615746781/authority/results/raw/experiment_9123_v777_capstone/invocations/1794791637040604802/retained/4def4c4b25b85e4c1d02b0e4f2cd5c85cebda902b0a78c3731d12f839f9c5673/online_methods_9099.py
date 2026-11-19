"""REQ-REPORT-9099-V776: freeze human rosters and qualify causal mechanics only."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from carnot.autoresearch import spline_basis_8986 as basis
from carnot.memory import human_online_9099 as learner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.human_methods_9084 import ref as ref, gate as gate
from carnot.reporting.natural_label_runner_8975 import read_ref
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_9099_v776_online_methods", "exp9099-online-methods"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_online_methods_9099.py"
OWNED = [
    "python/carnot/memory/human_online_9099.py",
    "python/carnot/reporting/online_methods_9099.py",
    "python/carnot/reporting/online_methods_runner_9099.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
AUTHORITY = "results/experiment_9097_v776_human_authority.json"
CONTRACT = "results/experiment_9096_v776_contract.json"
PROTOCOL = "openspec/change-proposals/v776-online-human-protocol.json"
DEFAULT_INPUT = Path("/tmp/carnot-9099-private")
PINS = {
    CONTRACT: (
        "sha256:71021947c3a5165e2575ae4611fc5d2b9108c5544c289ab9a84c8de37f0c1076",
        "contract_ready_score",
        1,
    ),
    AUTHORITY: (
        "sha256:24c9e0983d9a4eb0aa5a488b5467386e57d6f13b79447f0f3a2f3d1c68e25532",
        "source_ready_score",
        1,
    ),
    "results/experiment_8435_v726_capacity_benefit_audit.json": (
        "sha256:e2e58cbf680becad2f1572629337a58d747911492c24fa43697a8d867f4b0a51",
        "learning_audit_ready_score",
        1,
    ),
    "openspec/change-proposals/v726-methods-protocol.json": (
        "sha256:bb868af1d6f173c5b98155498e5daadd286e2641ef4aa3c20b7a0671b1b95117",
        "schema",
        "v726_methods_v1",
    ),
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual boundaries so supervision never mistakes a quiet child for progress."""
    print(f"[exp9099] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Read the separate frozen protocol without modifying the inherited static comparison."""
    return dict(json.loads((ROOT / PROTOCOL).read_bytes()))


def rosters(rows: list[Json]) -> Json:
    """Rank original qid strings and preserve each group's original three sibling positions."""
    groups: dict[str, list[Json]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["original_row"]):
        groups[r["qid"]].append(r)
    roles = {
        role: [q for q, siblings in groups.items() if siblings[0]["role"] == role]
        for role in ("TRAIN", "CAL", "TEST")
    }
    if [len(roles[r]) for r in ("TRAIN", "CAL", "TEST")] != [150, 50, 100] or any(
        len(rs) != 3
        or len({r["role"] for r in rs}) != 1
        or len({r["group_id"] for r in rs}) != 1
        or len({r["unit_id"] for r in rs}) != 3
        for rs in groups.values()
    ):
        raise ValueError("original_roster")
    train = sorted(
        roles["TRAIN"], key=lambda q: hashlib.sha256(("v776-online" + q).encode()).hexdigest()
    )
    donors = (
        np.random.default_rng(776099).choice(sorted(train[:100]), size=50, replace=True).tolist()
    )
    return dict(
        initial_group_roster=train[:100],
        stream_roster=[train[i : i + 5] for i in range(100, 150, 5)],
        guard_roster=roles["CAL"],
        retention_roster=roles["TEST"],
        donor_map=dict(zip(train[100:], donors, strict=True)),
        sibling_roster={q: [r["unit_id"] for r in rs] for q, rs in groups.items()},
        original_group_ids={q: rs[0]["group_id"] for q, rs in groups.items()},
    )


def authenticate(root: Path) -> Json:
    """Exact primary pins and original operand hashes precede every dependent fixture."""
    checks, references, values = [], [], {}
    for name, (pin, field, expected) in PINS.items():
        path = root / name
        references.append(ref(path))
        check = gate(
            path, "sha256", pin, ref(path)["sha256"] or "missing_file", "primary_bytes", name
        )
        checks.append(check)
        if check["passed"]:
            value = json.loads(path.read_bytes())
            values[name] = value
            checks.append(
                gate(
                    path, field, expected, value.get(field, "missing_field"), "upstream_field", name
                )
            )
    roster: Json = {}
    if all(c["passed"] for c in checks):
        source = values[AUTHORITY]
        inherited = values["openspec/change-proposals/v726-methods-protocol.json"]
        bound_refs = [
            *values[CONTRACT]["protocol_references"],
            source["evaluator_label_reference"],
            source["static_protocol_reference"],
            *source["label_authority_reference"]["files"].values(),
        ]
        for reference in bound_refs:
            path = Path(reference["path"])
            references.append(ref(path))
            checks.append(
                gate(
                    path,
                    "sha256",
                    reference["sha256"],
                    ref(path)["sha256"] or "missing_file",
                    "immutable_operand",
                    AUTHORITY,
                )
            )
        if all(c["passed"] for c in checks):
            roster = rosters(source["rows"])
            observed = {
                role: sorted({r["group_id"] for r in source["rows"] if r["role"] == role})
                for role in ("TRAIN", "CAL", "TEST")
            }
            expected_roles = {
                role: sorted(inherited["partition"]["roles"][old])
                for role, old in [("TRAIN", "fit"), ("CAL", "tune"), ("TEST", "test")]
            }
            checks.append(
                gate(
                    root / AUTHORITY,
                    "original_roles",
                    expected_roles,
                    observed,
                    "original_partition",
                    AUTHORITY,
                )
            )
    return dict(
        gate_check_summary=checks,
        source_references=references,
        roster=roster,
        blocker=next((c["check"] for c in checks if not c["passed"]), None),
    )


def fixture(condition: str) -> tuple[Json, Json]:
    """Balanced paired initialization has an exact zero optimum; synthetic streams are not human evidence."""
    x = np.repeat(np.random.default_rng(776099).normal(size=(50, 4)), 6, axis=0)
    fit = learner.fit_initial(x, np.tile([0, 0, 0, 1, 1, 1], 50))
    online = np.ones((15, 27))
    guard = np.zeros((150, 27))
    retention = np.zeros((300, 27))
    request = dict(
        initial=fit["initial"],
        initial_labels={f"private-init-{i}": [i % 2] * 3 for i in range(100)},
        donor_qids=["private-init-0"] * 5,
        online_design={},
        guard_design={},
        retention_design={},
        guard_labels=np.tile([0, 1], 75).tolist(),
        donor_labels=[[0, 0, 0]] * 5,
        online_qids=[f"private-online-{i}" for i in range(5)],
        retention_qids=[f"private-retention-{i}" for i in range(100)],
        condition="cpu_fixture_" + condition,
    )
    for kind in ("spline", "linear"):
        width = 217 if kind == "spline" else 28
        request["online_design"][kind] = (
            basis.design(online, kind == "spline").reshape(5, 3, width).tolist()
        )
        request["guard_design"][kind] = basis.design(guard, kind == "spline").tolist()
        request["retention_design"][kind] = (
            basis.design(retention, kind == "spline").reshape(100, 3, width).tolist()
        )
    ys = (
        [[1, 1, 1]] * 5
        if condition == "positive"
        else np.random.default_rng(776099).integers(0, 2, (5, 3)).tolist()
    )
    return request, dict(labels=ys)


def donor_changes(rows: list[Json], roster: Json) -> Json:
    """After map freeze, the evaluator counts changed original sibling labels without updating a learner."""
    labels = {r["unit_id"]: r["annotator1"] for r in rows}
    changes = []
    for qid, donor in roster["donor_map"].items():
        pairs = [
            (labels[a], labels[b])
            for a, b in zip(
                roster["sibling_roster"][qid], roster["sibling_roster"][donor], strict=True
            )
        ]
        observed = [(a, b) for a, b in pairs if a is not None and b is not None]
        count = sum(a != b for a, b in observed)
        changes.append(
            dict(
                original_qid=qid,
                donor_qid=donor,
                numerator=count,
                denominator=len(observed),
                status="completed" if len(observed) == 3 else "censored",
                censor_reason=None if len(observed) == 3 else "missing_original_A1",
            )
        )
    numerator, denominator = (
        sum(r["numerator"] for r in changes),
        sum(r["denominator"] for r in changes),
    )
    return dict(
        rows=changes,
        numerator=numerator,
        denominator=denominator,
        fraction=numerator / denominator if denominator else None,
        scope="descriptive exposed-label diagnostic after map freeze; not a learner input",
    )


def operands(condition: str) -> tuple[Json, Json]:
    """Freeze negative controls before they execute, including a tail-only truth mutation."""
    request, labels = fixture("no_signal" if condition == "no_signal" else "positive")
    if condition == "future_labels":
        labels["labels"][4] = [0, 0, 0]
    if condition == "sibling_labels":
        labels["labels"][0] = [0, 0, 0]
    if condition == "guard_reject":
        request["guard_labels"] = [0] * 150
    return request, labels


CONDITIONS = ("positive", "no_signal", "future_labels", "sibling_labels", "guard_reject")


def evaluate(raw: Path) -> Json:
    """Run constructed streams and real child exits; synthetic wins remain mechanics evidence."""
    for condition in CONDITIONS:
        progress("fixture_before_" + condition)
        request, labels = operands(condition)
        directory = raw / condition
        atomic_json(directory / "request.json", request)
        atomic_json(directory / "labels.json", labels)
        learner.drive(request, directory / "store", directory / "labels.json")
        progress("fixture_after_" + condition, 5, 0)
    receipts, resumes = [], []
    for point in protocol()["recovery"]["hard_exits"]:
        directory = raw / point
        prefix = [str(ROOT / ".venv/bin/python"), "-u"]
        if os.environ.get("COVERAGE_RCFILE"):
            prefix += ["-m", "coverage", "run", "--rcfile=" + os.environ["COVERAGE_RCFILE"]]
        argv = prefix + [
            str(ROOT / CLI),
            "--worker",
            str(raw / "positive/request.json"),
            str(directory),
            str(raw / "positive/labels.json"),
        ]
        receipt = child(
            point,
            argv + ["--crash", point],
            raw / "child_logs",
            expected=73,
            deadline=60,
            heartbeat=20,
        )
        receipts.append(receipt)
        if receipt["passed"]:
            atomic_json(
                directory / "interrupted.json", json.loads((directory / "state.json").read_bytes())
            )
        resumes.append(
            child(point + "_resume", argv, raw / "child_logs", deadline=60, heartbeat=20)
        )
    atomic_json(raw / "recovery_receipts.json", dict(hard=receipts, resumed=resumes))
    return reduce(raw)


def reduce(raw: Path) -> Json:
    """Reconstruct every trajectory and aggregate instead of trusting saved totals or rehashed state."""
    states = {}
    for condition in CONDITIONS:
        request, labels = operands(condition)
        directory = raw / condition
        if (
            json.loads((directory / "request.json").read_bytes()) != request
            or json.loads((directory / "labels.json").read_bytes()) != labels
        ):
            raise ValueError("fixture_operand_drift")
        with tempfile.TemporaryDirectory(prefix="carnot-9099-replay-") as replay:
            fresh = learner.drive(request, Path(replay), directory / "labels.json")
        envelope = json.loads((directory / "store/state.json").read_bytes())
        if fresh != envelope["state"] or canonical_hash(fresh) != envelope["sha256"]:
            raise ValueError("trajectory_drift")
        states[condition] = fresh
    positive = states["positive"]
    duplicate = json.loads(json.dumps(positive))
    no_second_effect = (
        not learner.feedback(duplicate, operands("positive")[0], 0, [1, 1, 1])
        and duplicate == positive
    )

    def donor(state: Json) -> list[Json]:
        """Donor predictions depend only on known initialization labels and public features."""
        return [r for r in state["rows"] if r["arm"] == "donor_label_spline"]

    rejected = states["guard_reject"]
    checks = dict(
        future_label=positive["rows"] == states["future_labels"]["rows"],
        sibling_leak=positive["rows"][:30] == states["sibling_labels"]["rows"][:30],
        duplicate_ack=no_second_effect,
        donor_causality=donor(positive) == donor(states["sibling_labels"]),
        rollback=all(
            not r["passed"] and r["before_hash"] == r["after_hash"]
            for r in rejected["rollback_checks"]
            if r["arm"] != "donor_label_spline"
        )
        and rejected["acknowledgments"] == list(range(5)),
    )
    rows, metrics = [], {}
    for condition in ("positive", "no_signal"):
        labels = operands(condition)[1]["labels"]
        scored = [
            dict(
                r,
                metric="Brier",
                p_error=r["value"],
                label=labels[r["group"]][r["sibling"]],
                value=(r["value"] - labels[r["group"]][r["sibling"]]) ** 2,
                absolute_metric=(r["value"] - labels[r["group"]][r["sibling"]]) ** 2,
                numerator=(r["value"] - labels[r["group"]][r["sibling"]]) ** 2,
            )
            for r in states[condition]["rows"]
        ]
        rows.extend(scored)
        metrics[condition] = {
            arm: float(np.mean([r["value"] for r in scored if r["arm"] == arm and r["group"] >= 1]))
            for arm in learner.ARMS
        }
    checks["positive_learning"] = (
        metrics["positive"]["adaptive_spline"] < metrics["positive"]["frozen_spline"]
    )
    checks["no_signal_fixture"] = len(states["no_signal"]["acknowledgments"]) == 5
    receipts = json.loads((raw / "recovery_receipts.json").read_bytes())
    exact = len(receipts["hard"]) == len(receipts["resumed"]) == 3
    for point, receipt, resumed in zip(
        protocol()["recovery"]["hard_exits"], receipts["hard"], receipts["resumed"], strict=True
    ):
        for entry in (receipt, resumed):
            for stream in ("stdout", "stderr"):
                if ref(Path(entry[stream + "_path"]))["sha256"] != entry[stream + "_sha256"]:
                    raise ValueError("hard_exit_log_drift")
        interrupted = json.loads((raw / point / "interrupted.json").read_bytes())
        checkpoint = interrupted["state"]
        exact &= bool(
            receipt["passed"]
            and receipt["normal_exit"]
            and receipt["exit_code"] == 73
            and not receipt["timed_out"]
            and resumed["passed"]
            and resumed["exit_code"] == 0
            and interrupted["sha256"] == canonical_hash(checkpoint)
            and checkpoint["query"] in (1, 2)
            and checkpoint["pending"]
            and json.loads((raw / point / "state.json").read_bytes())["state"] == positive
        )
    return dict(
        rows=rows,
        fixture_metrics=metrics,
        causal_controls=[dict(check=k, passed=bool(v)) for k, v in checks.items()],
        hard_exit_receipts=receipts["hard"],
        recovery_receipts=receipts["resumed"],
        exact_resume=bool(exact),
        rollback_checks=positive["rollback_checks"] + rejected["rollback_checks"],
    )
