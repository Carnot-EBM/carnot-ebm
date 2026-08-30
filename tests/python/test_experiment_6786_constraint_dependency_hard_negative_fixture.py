"""Focused tests for the exact constraint-dependency graph fixture.

Spec refs: REQ-VERIFY-6786 and SCENARIO-VERIFY-6786-*.
"""

from __future__ import annotations

from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_6786_constraint_dependency_hard_negative_fixture as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/verifiable-reasoning/spec.md"


@pytest.fixture(scope="module")
def panel() -> dict:
    """Load the frozen proof panel once because fixture generation is read-only."""

    return exp.load_json_object(REPO_ROOT / exp.SOURCE_PANEL_RELATIVE_PATH)


@pytest.fixture(scope="module")
def units(panel: dict) -> list[dict]:
    """Build the complete frozen unit manifest once for focused semantic tests."""

    return exp.build_units(panel)


@pytest.fixture(scope="module")
def rows(units: list[dict]) -> list[dict]:
    """Build both negative classes once for row-level checks."""

    return exp.build_rows(units)


def test_req_verify_6786_spec_declares_the_exact_fixture_contract() -> None:
    """REQ-VERIFY-6786 anchors the graph, split, feature, and replay contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-6786") : spec.index("### REQ-VERIFY-6769")]

    for marker in (
        "REQ-VERIFY-6786",
        "SCENARIO-VERIFY-6786-GRAPH",
        "SCENARIO-VERIFY-6786-NEGATIVES",
        "SCENARIO-VERIFY-6786-SPLITS",
        "SCENARIO-VERIFY-6786-FEATURES",
        "SCENARIO-VERIFY-6786-REPLAY",
        "SCENARIO-VERIFY-6786-BLOCKED",
        "96 unique bounded graph units",
        "complete_blocked_constraint_graph_fixture",
        "constraint_group_fixture_ready",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section or field in exp.STANDARD_ARTIFACT_FIELDS


def test_scenario_verify_6786_preconditions_require_each_exact_authority(
    panel: dict, tmp_path: Path
) -> None:
    """SCENARIO-VERIFY-6786-BLOCKED checks source, builders, enumeration, and families."""

    summary = exp.evaluate_preconditions(repo_root=REPO_ROOT)
    assert summary["all_passed"] is True
    assert [row["check"] for row in summary["checks"]] == [
        "exp6768_artifact_exists",
        "exp6768_panel_ready",
        "exp6768_row_count",
        "exp6768_row_hashes",
        "declarative_constraint_group_builders",
        "cpu_exact_enumeration_path",
        "minimum_topology_families",
    ]

    missing = exp.evaluate_preconditions(
        repo_root=REPO_ROOT,
        source_panel_path=tmp_path / "missing.json",
    )
    assert missing["all_passed"] is False
    assert exp.first_failed_check(missing)["check"] == "exp6768_artifact_exists"

    broken_panel = deepcopy(panel)
    broken_panel["rows"][0]["row_sha256"] = "sha256:broken"
    broken_path = tmp_path / "broken.json"
    broken_path.write_text(json.dumps(broken_panel), encoding="utf-8")
    broken = exp.evaluate_preconditions(repo_root=REPO_ROOT, source_panel_path=broken_path)
    assert exp.first_failed_check(broken)["check"] == "exp6768_row_hashes"

    unavailable = exp.evaluate_preconditions(
        repo_root=REPO_ROOT,
        group_builders=(lambda: (), lambda: ()),
        exact_enumerator=lambda _unit: {"enumerated_state_count": 0, "exact_assignments": []},
        topology_specs=exp.TOPOLOGY_SPECS[:2],
    )
    assert {row["check"] for row in unavailable["checks"] if not row["passed"]} == {
        "declarative_constraint_group_builders",
        "cpu_exact_enumeration_path",
        "minimum_topology_families",
    }


def test_scenario_verify_6786_graph_identity_groups_and_enumeration_are_exact(
    units: list[dict],
) -> None:
    """SCENARIO-VERIFY-6786-GRAPH freezes unique ordered exact-enumerable graphs."""

    assert len(units) == exp.UNIT_COUNT == 96
    assert len({unit["unit_id"] for unit in units}) == 96
    assert len({unit["graph_id"] for unit in units}) == 96
    assert {unit["unit_role"] for unit in units} == {"satisfiable"}
    assert all(unit["exact_assignments"] for unit in units)
    assert all(unit["contradiction_certificate"] is None for unit in units)

    for unit in units:
        assert unit["graph_serialization"] == exp.canonical_json(unit["graph"])
        assert unit["graph_id"] == exp.sha256_json(unit["graph"])
        groups = unit["graph"]["local_groups"]
        edges = unit["graph"]["dependency_edges"]
        assert [group["group_id"] for group in groups] == sorted(
            group["group_id"] for group in groups
        )
        assert all(group["group_type"] == "one_hot_domain" for group in groups)
        assert all(edge["dependency_id"] for edge in edges)
        replay = exp.enumerate_exact_semantics(unit)
        assert replay["exact_assignments"] == unit["exact_assignments"]
        assert replay["enumerated_state_count"] == 2 ** len(groups)
        assert unit["provenance"]["source_panel_row_sha256"].startswith("sha256:")


def test_scenario_verify_6786_hard_and_easy_negatives_preserve_failure_semantics(
    units: list[dict], rows: list[dict]
) -> None:
    """SCENARIO-VERIFY-6786-NEGATIVES separates dependency failures from local shortcuts."""

    assert len(rows) == exp.ROW_COUNT == 192
    assert len({row["row_id"] for row in rows}) == 192
    unit_by_id = {unit["unit_id"]: unit for unit in units}
    for row in rows:
        receipt = exp.evaluate_candidate(unit_by_id[row["unit_id"]], row["candidate_assignment"])
        assert receipt == row["exact_receipt"]
        assert row["exact_valid"] is False
        assert row["row_sha256"] == exp.row_checksum(row)
        if row["negative_class"] == "hard_cross_dependency_failure":
            assert receipt["local_checks_passed"] is True
            assert receipt["failed_local_group_ids"] == []
            assert len(receipt["failed_dependency_ids"]) == 1
            assert row["named_broken_dependency"] == receipt["failed_dependency_ids"][0]
        else:
            assert row["negative_class"] == "easy_local_failure"
            assert receipt["local_checks_passed"] is False
            assert len(receipt["failed_local_group_ids"]) == 1
            assert row["named_broken_local_group"] == receipt["failed_local_group_ids"][0]


def test_scenario_verify_6786_splits_are_disjoint_by_topology(
    units: list[dict], rows: list[dict]
) -> None:
    """SCENARIO-VERIFY-6786-SPLITS holds out whole topology families."""

    split_receipt = exp.summarize_splits(units, rows)
    family_sets = [set(value["topology_families"]) for value in split_receipt.values()]

    assert set(split_receipt) == {"train", "development", "held_topology_test"}
    assert all(len(families) == 1 for families in family_sets)
    assert family_sets[0].isdisjoint(family_sets[1])
    assert family_sets[0].isdisjoint(family_sets[2])
    assert family_sets[1].isdisjoint(family_sets[2])
    assert all(value["unit_count"] == 32 for value in split_receipt.values())
    assert all(value["negative_row_count"] == 64 for value in split_receipt.values())


def test_scenario_verify_6786_feature_projection_obeys_allowlist_and_denylist(
    rows: list[dict],
) -> None:
    """SCENARIO-VERIFY-6786-FEATURES keeps oracle data outside proposal features."""

    assert exp.audit_feature_contract(rows) == []
    assert all(set(row["proposal_features"]) == set(exp.FEATURE_ALLOWLIST) for row in rows)
    assert set(exp.REQUIRED_DENIED_FEATURES) <= set(exp.FEATURE_DENYLIST)

    changed = deepcopy(rows[:1])
    changed[0]["proposal_features"]["exact_valid"] = False
    violations = exp.audit_feature_contract(changed)
    assert violations == [f"{changed[0]['row_id']}.exact_valid"]


def test_scenario_verify_6786_generation_and_cold_replay_are_deterministic(
    panel: dict,
    units: list[dict],
    rows: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-VERIFY-6786-REPLAY recomputes all 192 labels in a fresh process."""

    second_units = exp.build_units(panel)
    second_rows = exp.build_rows(second_units)
    assert second_units == units
    assert second_rows == rows

    replay = exp.run_cold_replay(units, rows, repo_root=REPO_ROOT)
    assert replay["agreement"] is True
    assert replay["fresh_process"] is True
    assert replay["replayed_row_count"] == 192
    assert replay["mismatches"] == []
    assert replay["cold_pid"] != replay["producer_pid"]

    changed = deepcopy(rows[:1])
    changed[0]["exact_valid"] = True
    direct = exp.replay_payload(units[:1], changed)
    assert direct["agreement"] is False
    assert direct["mismatches"] == [changed[0]["row_id"]]

    worker_payload = json.dumps({"units": units[:1], "rows": rows[:2]})
    monkeypatch.setattr(exp.sys, "stdin", StringIO(worker_payload))
    assert exp._cold_replay_worker() == 0
    worker_receipt = json.loads(capsys.readouterr().out)
    assert worker_receipt["agreement"] is True
    assert worker_receipt["replayed_row_count"] == 2

    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=7, stderr="forced failure"),
    )
    with pytest.raises(RuntimeError, match="forced failure"):
        exp.run_cold_replay(units[:1], rows[:2], repo_root=REPO_ROOT)


def test_req_verify_6786_artifact_is_complete_and_row_derived(tmp_path: Path) -> None:
    """REQ-VERIFY-6786 emits every required field and a stable ready gate."""

    artifact_path = tmp_path / "fixture.json"
    artifact = exp.write_outputs(
        run_date="20260830",
        artifact_path=artifact_path,
        repo_root=REPO_ROOT,
        duration_s=1.25,
    )
    loaded = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert exp.validate_artifact(artifact) == []
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["constraint_group_fixture_ready"] is True
    assert artifact["duplicate_rows"] == 0
    assert artifact["future_feature_violations"] == []
    assert artifact["cold_replay_agreement"]["agreement"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"] == exp.reproducibility_checksum(artifact)
    assert all(
        count == 32 for count in artifact["local_pass_cross_dependency_fail_counts"].values()
    )
    assert all(count == 32 for count in artifact["easy_negative_counts"].values())

    broken = deepcopy(artifact)
    broken["constraint_group_fixture_ready"] = False
    assert "ready artifact must set readiness true" in exp.validate_artifact(broken)
    broken = deepcopy(artifact)
    broken["reproducibility_checksum"] = "bad"
    assert "reproducibility checksum mismatch" in exp.validate_artifact(broken)

    validation_cases = (
        (lambda value: value.pop("schema"), "required field set mismatch"),
        (
            lambda value: value["field_principles"].pop("schema"),
            "field principle coverage mismatch",
        ),
        (
            lambda value: value.__setitem__("inference_substrate", "bad"),
            "inference substrate mismatch",
        ),
        (lambda value: value.__setitem__("duration_s", -1), "duration_s must be non-negative"),
        (lambda value: value.__setitem__("random_seed", -1), "random seed mismatch"),
        (
            lambda value: value.__setitem__("verdict_class", "unknown"),
            "verdict class is outside the closed enum",
        ),
        (
            lambda value: value.__setitem__("honest_verdict", "not terminal"),
            "honest verdict lacks a terminal prefix",
        ),
        (lambda value: value.__setitem__("status", "bad"), "ready artifact status mismatch"),
        (
            lambda value: value["frozen_manifest"].__setitem__("units", []),
            "ready manifest must contain 96 units",
        ),
        (lambda value: value.__setitem__("rows", []), "ready artifact must contain 192 rows"),
        (
            lambda value: value.__setitem__("duplicate_rows", 1),
            "ready artifact contains duplicate rows",
        ),
        (
            lambda value: value.__setitem__("future_feature_violations", ["bad"]),
            "ready artifact contains forbidden proposal features",
        ),
        (
            lambda value: value["cold_replay_agreement"].__setitem__("agreement", False),
            "ready artifact lacks cold replay agreement",
        ),
        (
            lambda value: value["gate_check_summary"].__setitem__("all_passed", False),
            "ready artifact has failed gates",
        ),
        (
            lambda value: value.__setitem__("verifier_is_oracle", True),
            "verifier_is_oracle must remain false",
        ),
    )
    for mutate, expected_error in validation_cases:
        changed = deepcopy(artifact)
        mutate(changed)
        assert expected_error in exp.validate_artifact(changed)


def test_scenario_verify_6786_blocked_artifact_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-6786-BLOCKED writes a complete block with no generated rows."""

    artifact = exp.build_artifact(
        run_date="20260830",
        repo_root=REPO_ROOT,
        source_panel_path=tmp_path / "missing.json",
        duration_s=0.1,
    )

    assert exp.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_blocked_constraint_graph_fixture"
    assert artifact["rows"] == []
    assert artifact["constraint_group_fixture_ready"] is False
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_constraint_graph_fixture")
    assert artifact["gate_check_summary"]["first_failure"]["check"] == "exp6768_artifact_exists"

    with pytest.raises(ValueError, match="YYYYMMDD"):
        exp.build_artifact(run_date="2026-08-30", repo_root=REPO_ROOT)

    relative = exp.write_outputs(
        run_date="20260830",
        artifact_path=Path("relative-blocked.json"),
        repo_root=tmp_path,
        source_panel_path=tmp_path / "missing.json",
        duration_s=0.1,
    )
    assert relative["constraint_group_fixture_ready"] is False
    assert (tmp_path / "relative-blocked.json").is_file()

    monkeypatch.setattr(
        exp,
        "run_cold_replay",
        lambda *_args, **_kwargs: {
            "agreement": False,
            "replayed_row_count": exp.ROW_COUNT,
            "mismatches": ["forced"],
            "rows_sha256": exp.sha256_json([]),
        },
    )
    internal_block = exp.build_artifact(
        run_date="20260830",
        repo_root=REPO_ROOT,
        duration_s=0.1,
    )
    assert internal_block["status"] == "complete_blocked_constraint_graph_fixture"
    assert internal_block["gate_check_summary"]["first_failure"]["check"] == (
        "cold_replay_agreement"
    )


def test_req_verify_6786_defensive_inputs_fail_closed(tmp_path: Path, units: list[dict]) -> None:
    """REQ-VERIFY-6786 rejects malformed roots, unknown semantics, and short panels."""

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON root"):
        exp.load_json_object(non_object)

    assert exp.first_failed_check(exp._summary([]))["check"] == "all_preconditions"
    with pytest.raises(ValueError, match="unknown topology"):
        exp._topology_edge_pairs("unknown", 3)
    with pytest.raises(ValueError, match="unknown dependency"):
        exp._dependency_passes("unknown", 0, 0)
    with pytest.raises(ValueError, match="fewer than 96"):
        exp.build_units({"rows": []})

    no_edge_unit = deepcopy(units[0])
    no_edge_unit["graph"]["dependency_edges"] = []
    with pytest.raises(ValueError, match="no single-dependency"):
        exp._hard_negative(no_edge_unit)

    nested = exp.build_rows(units[:1])[:1]
    nested[0]["proposal_features"]["local_groups"][0]["exact_valid"] = False
    assert exp.audit_feature_contract(nested) == [
        f"{nested[0]['row_id']}.local_groups[0].exact_valid"
    ]
