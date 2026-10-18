"""REQ-REPORT-8555: count primitive operations without hiding true equalities.

Assignments record attempted coefficient stores. State comparisons record actual
changes. Keeping both prevents an update-event count from posing as parameter
activity, and keeps aliases from becoming additional scientific measurements.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting import historical_learning_8554 as historical
from carnot.reporting import v733_update_metrics as shipped
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import objective_derivatives_8461 as derivatives

Json = dict[str, Any]
__all__ = ["historical"]
ROOT = historical.ROOT
NAME = "experiment_8555_v735_canonical_update_accounting"
TASK = "exp8555-canonical-update-accounting"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [
    "tests/python/test_canonical_update_8555.py",
    "tests/python/test_canonical_update_runner_8555.py",
]
OWNED = [
    f"python/carnot/reporting/canonical_update{s}_8555.py" for s in ("", "_report", "_runner")
] + [CLI]
MODEL_SPECS: list[Json] = []
HISTORICAL_NAME = historical.NAME
HISTORICAL_PIN = "sha256:11366167c60a52398a2d0a4d88c9fbfe879292b445492990438985adb081e9bd"
ALIAS_MAP = {
    "label_updates": "applied_update_events",
    "coefficient_change_count": "changed_coordinate_count",
}
DEFINITIONS = dict(
    applied_update_events="Committed numerical update calls; frozen releases excluded.",
    physical_coordinate_writes="Sparse assignment indices counted with multiplicity, including zero gradients.",
    changed_coordinate_count="Unequal stored parameter coordinates summed over committed events.",
    decay_scale_changes="Changes of the shared effective-coefficient multiplier.",
    effective_coefficient_changes="Unequal parameter times decay_scale coordinates, summed per event.",
    unique_changed_coordinates="Union of changed coordinate identities in seed/arm namespaces.",
    admission_events="Admitted pending records whose predictions preceded label release.",
    label_release_events="Released admitted labels, including labels on frozen arms.",
)
GATE_FIELDS = {"check", "upstream", "path", "sha256", "field", "operator", "expected", "observed"}
reference = historical.old.reference
read = historical.old.read


def checksum(value: Json) -> str:
    """Exclude the checksum itself so replay can bind the complete numeric representation."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def gate(path: Path, field: str, expected: Any, observed: Any, *, upstream: str = TASK) -> Json:
    """Keep missing evidence as null so it cannot be mistaken for a measured zero."""
    return dict(
        check=field,
        upstream=upstream,
        path=str(path),
        sha256=sha256_file(path) if path.is_file() else None,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def authenticate(root: Path) -> tuple[Json, list[Json], list[Json]]:
    """Pin the qualified historical producer and every source/test byte it actually used."""
    path = root / "results" / (HISTORICAL_NAME + ".json")
    refs: list[Json] = []
    value: Json = {}
    try:
        value = read(dict(path=str(path), sha256=HISTORICAL_PIN))
        refs = [reference(path), *value["source_artifact_hashes"], *value["code_config_hashes"]]
        for ref in refs:
            if reference(Path(ref["path"])) != ref:
                raise ValueError("historical_source_test_hash")
        terminal = read(reference(Path(value["terminal_validation_sidecar_path"])))
        bound = read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))
        if not bound["report"]["passed"] or not historical.replay(path):
            raise ValueError("historical_bound_replay")
        failures = [
            gate(path, field, expected, value.get(field), upstream=historical.TASK)
            for field, expected in [
                ("task_id", historical.TASK),
                ("historical_learning_validation_ready_score", 1),
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ]
            if value.get(field) != expected
        ]
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures = [
            gate(
                path,
                "historical_source_test_authentication",
                True,
                None if isinstance(error, FileNotFoundError) else str(error),
                upstream=historical.TASK,
            )
        ]
    return value, refs, failures


def reduce(rows: list[Json]) -> Json:
    """Use raw array comparisons and assignment lists, independently of saved totals."""
    result = dict.fromkeys(DEFINITIONS, 0)
    seen: set[str] = set()
    coordinates: set[str] = set()
    for row in rows:
        identity = row["namespace"] + ":" + row["event_id"]
        if identity in seen:
            raise ValueError("duplicate_event")
        seen.add(identity)
        before, after, writes = row["before"], row["after"], row["assignment_indices"]
        shipped.coordinate_row(before, after, writes, event=identity, namespace=row["namespace"])
        if not row["applied"] and (before != after or writes):
            raise ValueError("inactive_update")
        left, right = before["parameters"], after["parameters"]
        changed = [i for i, (a, b) in enumerate(zip(left, right, strict=True)) if a != b]
        effective = sum(
            a * before["decay_scale"] != b * after["decay_scale"]
            for a, b in zip(left, right, strict=True)
        )
        result["applied_update_events"] += int(row["applied"])
        result["physical_coordinate_writes"] += len(writes)
        result["changed_coordinate_count"] += len(changed)
        result["decay_scale_changes"] += int(before["decay_scale"] != after["decay_scale"])
        result["effective_coefficient_changes"] += effective
        result["admission_events"] += int(row["admitted"])
        result["label_release_events"] += int(row["label_released"])
        coordinates.update(f"{row['namespace']}:{i}" for i in changed)
    result["unique_changed_coordinates"] = len(coordinates)
    return result


def visibility(records: list[Json], metrics: Json) -> Json:
    """Require original metric names and exact numeric values in validator-visible metrics."""
    return dict(
        passed=bool(records)
        and set(metrics) == {r["name"] for r in records}
        and all(
            type(metrics.get(r["name"])) in (int, float) and metrics[r["name"]] == r["value"]
            for r in records
        ),
        canonical_names=[r["name"] for r in records],
        projected_names=list(metrics),
        representation="unchanged adversarial _flatten_metrics(metrics)",
    )


def canonical(rows: list[Json]) -> tuple[list[Json], Json, Json, list[Json]]:
    """Cross-check independent reductions before exposing all distinct quantities unchanged."""
    totals = reduce(rows)
    observed = shipped.aggregate([r for r in rows if r["applied"]])
    observed.update(
        admission_events=sum(bool(r["admitted"]) for r in rows),
        label_release_events=sum(bool(r["label_released"]) for r in rows),
    )
    checks = observed == totals
    records = []
    for name, definition in DEFINITIONS.items():
        sources = []
        for row in rows:
            if reduce([row])[name]:
                sources.append(row["namespace"] + ":" + row["event_id"])
        records.append(
            dict(
                name=name,
                value=totals[name],
                unit="events"
                if name.endswith("events")
                else "assignments"
                if name == "physical_coordinate_writes"
                else "coordinates"
                if "coordinate" in name
                else "changes",
                definition=definition,
                source_event_ids=sources,
            )
        )
    receipt = visibility(records, totals)
    receipt["passed"] = bool(receipt["passed"] and checks and rows)
    receipt["raw_reduction_matches_shipped"] = checks
    relations = [
        dict(
            left=left,
            right=right,
            left_value=totals[left],
            right_value=totals[right],
            relation="equal" if totals[left] == totals[right] else "different",
            derivation=derivation,
            independently_recomputed=checks,
            difference=totals[left] - totals[right],
        )
        for left, right, derivation in [
            (
                "applied_update_events",
                "decay_scale_changes",
                "sum(applied) versus sum(before.scale != after.scale); equality requires one changed scale per applied event",
            ),
            (
                "admission_events",
                "label_release_events",
                "sum(admitted) versus sum(label_released); original sealed roster has exactly one released label per admitted record",
            ),
            (
                "physical_coordinate_writes",
                "changed_coordinate_count",
                "sum(len(assignment_indices)) versus sum(before[i] != after[i]); duplicates and zero gradients break equality",
            ),
            (
                "changed_coordinate_count",
                "unique_changed_coordinates",
                "sum(event changed coordinates) versus cardinality(union(namespace,coordinate)); repeated changes do not add unique coordinates",
            ),
            (
                "effective_coefficient_changes",
                "changed_coordinate_count",
                "sum(before[i]*before.scale != after[i]*after.scale) versus stored changes; lazy decay changes unwritten coefficients",
            ),
        ]
    ]
    for row in rows:
        if row["namespace"] == "constructed_control":
            single = reduce([row])
            left, right = single["physical_coordinate_writes"], single["changed_coordinate_count"]
            relations.append(
                dict(
                    event_id=row["event_id"],
                    left="physical_coordinate_writes",
                    right="changed_coordinate_count",
                    left_value=left,
                    right_value=right,
                    relation="equal" if left == right else "different",
                    difference=left - right,
                    derivation="len(actual assignment list) versus number of unequal stored coordinates for this event; one_to_one assigns one previously unequal coordinate once",
                    independently_recomputed=checks,
                )
            )
    return records, totals, receipt, relations


def controls() -> list[Json]:
    """Execute fixed assignments so both unequal quantities and one-to-one identities are proved."""
    rows = []
    for name, assignments, scale, applied, admitted, released in [
        ("duplicate_assignment", [(0, 2.0), (0, 2.0), (1, 2.0)], 1.0, True, True, True),
        ("zero_gradient", [(0, 1.0), (1, 2.0)], 1.0, True, True, True),
        ("decay_only", [], 0.9, True, True, True),
        ("one_to_one", [(0, 2.0)], 1.0, True, True, True),
        ("frozen_arm", [], 1.0, False, True, True),
        ("pending_feedback", [], 1.0, False, True, False),
        ("rejected_feedback", [], 1.0, False, False, False),
    ]:
        before: Json = dict(parameters=[1.0, 2.0, 0.0], decay_scale=1.0)
        after = deepcopy(before)
        for index, value in assignments:
            after["parameters"][index] = value
        after["decay_scale"] = scale
        row = shipped.coordinate_row(
            before, after, [i for i, _ in assignments], event=name, namespace="constructed_control"
        )
        row.update(
            applied=applied,
            applied_update_events=int(applied),
            admitted=admitted,
            label_released=released,
            source_reference="fixed executed control operands",
        )
        rows.append(row)
    return rows


def history() -> tuple[list[Json], list[Json]]:
    """Recompute Exp8517 primitive transitions; the disqualified Exp8530 totals never enter."""
    evidence = shipped.historical_accounting()
    if not evidence["reconciliation_passed"]:
        raise ValueError("original_transition_reconciliation")
    indexed = {(r["namespace"], r["event_id"]): r for r in evidence["rows"]}
    primary = read(
        dict(
            path=str(ROOT / "results" / (shipped.historical.NAME + ".json")),
            sha256=shipped.HISTORICAL_PIN,
        )
    )
    work = read(primary["primitive_reference"])
    rows = []
    for cell in work["cells"]:
        cfg = read(cell["configuration"])
        for ref in cell["steps"]:
            step = read(ref)
            for record, trace in zip(step["sealed_state"]["pending"], step["updates"], strict=True):
                identity = trace["feedback_id"]
                row = deepcopy(indexed.get((cell["identity"], identity)))
                if row is None:
                    if not cfg["frozen"]:
                        raise ValueError("missing_applied_transition")
                    head = step["sealed_state"]["head"]
                    row = shipped.coordinate_row(
                        head, head, [], event=identity, namespace=cell["identity"]
                    )
                row.update(
                    applied=not cfg["frozen"],
                    applied_update_events=int(not cfg["frozen"]),
                    admitted=True,
                    label_released=True,
                    source_reference=ref,
                    admission_record=record,
                    label_release_record=trace,
                    seed=cell["seed"],
                    arm=cell["arm"],
                )
                rows.append(row)
    return rows, evidence["source_refs"]


def numerical(scratch: Path) -> list[Json]:
    """Reuse independent dense finite differences and genuine store barriers at original tolerances."""
    adapter = derivatives.adapter
    head = adapter.legacy.head()
    rows = [dict(row, name=row["unit_id"]) for row in derivatives.checks(head)]
    cfg = adapter.config(head, 8, "uniform", frozen=True)
    store = adapter.Store(scratch / "frozen", cfg)
    store.initialize()
    store.seal(0, adapter.legacy.questions(0))
    pending = store.read()
    labels = adapter.legacy.feedback(pending)
    try:
        store.release(0, dict(labels, future_label=1))
        rejected = False
    except ValueError as error:
        rejected = str(error) == "release_roster" and store.read() == pending
    store.release(0, labels)
    frozen = store.read()
    x = np.asarray([-10.0] * 9)
    changed, _ = adapter.update(head, x, 1)
    numeric_error = float(
        np.max(
            np.abs(
                adapter.sparse.parameters(changed)
                - (adapter.sparse.parameters(head) - 0.01 * derivatives.dense(head, x, 1))
            )
        )
    )
    rows += [
        dict(
            name="clipping",
            passed=numeric_error <= 1e-12,
            absolute_error=numeric_error,
            input=x.tolist(),
        ),
        dict(
            name="pending_feedback",
            passed=bool(pending["pending"]) and pending["head"] == head,
            before=head,
            sealed_state=pending,
        ),
        dict(name="rejected_feedback", passed=rejected, before_state_hash=canonical_hash(pending)),
        dict(
            name="frozen_arm",
            passed=frozen["head"] == head and frozen["parameter_updates"] == 0,
            after_state=frozen,
        ),
    ]
    return rows


def replay(path: Path) -> bool:
    """Rebuild primitive rows from original seals, so self-consistent envelope repairs still fail."""
    from carnot.reporting.canonical_update_report_8555 import build

    try:
        value = json.loads(path.read_bytes())
        if value["reproducibility_checksum"] != checksum(value):
            return False
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            if reference(Path(ref["path"])) != ref:
                return False
        work = read(value["primitive_reference"])
        for row in work["receipts"]:
            saved = json.loads(Path(row["stdout_path"]).with_suffix(".receipt.json").read_bytes())
            if row != saved:
                return False
        if (
            not work["private_control"]
            and work["historical"].get("historical_learning_validation_ready_score") == 1
        ):
            original_nodes = historical.original()[1]["nodes"]
            if set(work["collected_node_ids"]) != set(original_nodes) or work[
                "union"
            ] != historical.exact_union(
                original_nodes, [r for r in work["receipts"] if r.get("collected_node_ids")]
            ):
                return False
        rows = json.loads(Path(value["event_rows_reference"]["path"]).read_bytes())
        original = controls() if work["private_control"] else history()[0]
        return bool(rows == original and value == build(work, Path(value["output_path"])))
    except (OSError, ValueError, KeyError, TypeError):
        return False
