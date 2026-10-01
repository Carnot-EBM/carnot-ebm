"""Role custody and permitted access checks for REQ-VERIFY-7968."""

from copy import deepcopy

import pytest

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.source_boundary_7892 import EIGHT_ROLES
from carnot.verify import response_role_targets_7968 as roles
from carnot.verify import response_targets_7955 as targets
from test_experiment_7942_v689_sentence_labels import fixture as original_fixture


def fixture(path):
    """Keep the entire allocation while giving each family distinct source bytes."""
    original = original_fixture(path)
    data = {key: [] for key in original}
    index = 0
    for role, count in EIGHT_ROLES.items():
        for _ in range(count):
            source = f"Café source {index}."
            for key, values in original.items():
                row = deepcopy(values[index % 64])
                if "family_id" in row:
                    row["family_id"] = f"family-{index}"
                if "role" in row:
                    row["role"] = role
                if "source_cluster_id" in row:
                    row["source_cluster_id"] = roles.digest(source.encode())
                if "source_bytes" in row:
                    row["source_bytes"] = source.encode().hex()
                for field in ("source_id", "id", "response_id"):
                    if field in row:
                        row[field] = str(index)
                if "source_info" in row:
                    row["source_info"] = source
                data[key].append(row)
            index += 1
    atomic_json(path, data)
    return data


def joined(data):
    """Use the existing annotation join as the only label engine."""
    public, audit = targets.freeze(data["public"])
    rows, annotations = targets.join(public, audit, data)
    return public, audit, rows, annotations


def test_all_roles_counts_unknown_and_unicode(tmp_path):
    """SCENARIO-VERIFY-7968-CUSTODY: keep slots, unknowns and UTF-8 spans."""
    data = fixture(tmp_path / "input.json")
    data["responses"][0]["labels"] = None
    data["responses"][1]["quality"] = "truncated"
    data["responses"][2].update(
        response="Café.",
        labels=[
            dict(
                start=3,
                end=4,
                text="é",
                implicit_true=True,
                due_to_null=True,
                label_type="human",
                meta="fixture",
            )
        ],
    )
    data["public"][2]["answer_bytes"] = "Café.".encode().hex()
    public, audit, rows, annotations = joined(data)
    views, reduced = roles.partition(public, audit, data["roles"], rows, annotations)
    assert reduced["role_counts"] == EIGHT_ROLES
    assert reduced["sample_size_budget"]["intended"] == 640
    assert reduced["cross_role_overlap_count"] == 0
    assert reduced["class_counts_by_role"]["fit"]["unknown"] == 2
    assert reduced["response_roles_ready_score"] == 1
    assert rows[2]["y"] == 1 and rows[2]["implicit_true_excluded_y"] == 0
    assert next(r for r in annotations if r["family_id"] == "family-2")["end_byte"] == 5
    assert set(views["fit"]["public"]) == {"role", "request_rows", "boundaries"}
    assert len(views["evaluation"]["evaluator"]["rows"]) == 64
    # Class imbalance remains a downstream measurement concern.
    for response in data["responses"]:
        response.update(quality="good", labels=[])
    assert (
        roles.partition(*joined(data)[:2], data["roles"], *joined(data)[2:])[1][
            "response_roles_ready_score"
        ]
        == 1
    )


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("public", "public_fields"),
        ("roster", "original_role_roster"),
        ("swapped", "role_drift"),
        ("duplicate", "source_cluster_role"),
        ("normalized", "normalized_source_role"),
        ("row", "union_drift"),
    ],
)
def test_rejects_role_and_public_drift(tmp_path, mutation, reason):
    """SCENARIO-VERIFY-7968-CUSTODY: labels cannot change cohort membership."""
    data = fixture(tmp_path / "input.json")
    public, audit, rows, annotations = joined(data)
    if mutation == "public":
        public[0]["annotation_order"] = [1]
    if mutation == "roster":
        data["roles"][0]["role"] = "unknown"
    if mutation == "swapped":
        rows[0]["role"] = "evaluation"
    if mutation in {"duplicate", "normalized"}:
        other = next(i for i, row in enumerate(rows) if row["role"] != rows[0]["role"])
        source = bytes.fromhex(public[0]["source_bytes"])
        if mutation == "normalized":
            source = b" " + source + b"  "
        public[other]["source_bytes"] = source.hex()
        data["roles"][other]["source_cluster_id"] = roles.digest(source)
    if mutation == "row":
        rows[0]["response_sha256"] = "wrong"
    with pytest.raises(ValueError, match=reason):
        roles.partition(public, audit, data["roles"], rows, annotations)


def test_access_roles_and_hash_bound_seals(tmp_path):
    """SCENARIO-VERIFY-7968-ACCESS: labels require the correct consumer stage."""
    for role in EIGHT_ROLES:
        roles.authorize(role, "capture", {})
    for role, stage in [
        ("fit", "fitting"),
        ("tune", "fitting"),
        ("policy_design", "threshold_design"),
    ]:
        roles.authorize(role, stage, {})
    for role, stage in [
        ("evaluation", "fitting"),
        ("retention", "threshold_design"),
        ("fit", "evaluation"),
        ("bad", "capture"),
    ]:
        with pytest.raises(ValueError, match="role_access"):
            roles.authorize(role, stage, {})
    with pytest.raises(ValueError, match="seals_required"):
        roles.authorize("evaluation", "evaluation", {})
    seal = tmp_path / "sealed.json"
    atomic_json(seal, {"immutable": True})
    seals = {key: dict(path=str(seal), sha256=sha256_file(seal)) for key in ("heads", "policies")}
    for role in EIGHT_ROLES.keys() - {"fit", "tune", "policy_design"}:
        roles.authorize(role, "evaluation", seals)
    seals["heads"]["sha256"] = "bad"
    with pytest.raises(ValueError, match="hash_drift"):
        roles.authorize("evaluation", "evaluation", seals)


def test_union_roster_and_capture_boundary_checks(tmp_path):
    """REQ-VERIFY-7968: bound views reject missing slots and altered metadata."""
    data = fixture(tmp_path / "input.json")
    public, audit, rows, annotations = joined(data)
    with pytest.raises(ValueError, match="union_drift"):
        roles.partition(public, audit, data["roles"], rows[:-1], annotations)
    views, _ = roles.partition(public, audit, data["roles"], rows, annotations)
    path = tmp_path / "view.json"
    atomic_json(path, views["fit"]["evaluator"])
    assert roles.read_view(path, sha256_file(path), "fit", "fitting", {})
    view = deepcopy(views["fit"]["evaluator"])
    view["rows"][0]["role"] = "evaluation"
    atomic_json(path, view)
    with pytest.raises(ValueError, match="role_drift"):
        roles.read_view(path, sha256_file(path), "fit", "fitting", {})
    view = deepcopy(views["fit"]["public"])
    view["boundaries"] = []
    atomic_json(path, view)
    with pytest.raises(ValueError, match="public_drift"):
        roles.read_view(path, sha256_file(path), "fit", "capture", {})
