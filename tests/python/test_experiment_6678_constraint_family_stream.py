"""Tests for the Exp6678 independent constraint-family stream.

Spec refs: REQ-LEARN-6678, SCENARIO-LEARN-6678-PREQUENTIAL,
SCENARIO-LEARN-6678-FAMILY-BLIND, SCENARIO-LEARN-6678-EXACT,
SCENARIO-LEARN-6678-RETIREMENT, SCENARIO-LEARN-6678-ISOLATION,
SCENARIO-LEARN-6678-RESTART, SCENARIO-LEARN-6678-ROLLBACK,
SCENARIO-LEARN-6678-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from carnot import experiment_6678_constraint_family_stream as mod


REPO = Path(__file__).resolve().parents[2]
PASSING_TESTS = [
    {"command": command, "exit_code": 0, "summary": "passed"}
    for command in mod.VERIFICATION_COMMANDS
]


def _artifact(tmp_path: Path, *, write: bool = False) -> dict[str, object]:
    return mod.build_artifact(
        repo_root=REPO,
        output_path=tmp_path / mod.RESULT_PATH.name,
        state_path=tmp_path / "state",
        date="20260827",
        duration_s=1.0,
        tests_run=PASSING_TESTS,
        write=write,
    )


def test_req_learn_6678_spec_declares_complete_contract() -> None:
    """REQ-LEARN-6678: OpenSpec owns all stream safety boundaries."""

    text = (REPO / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-6678") :]
    for marker in (
        "SCENARIO-LEARN-6678-PREQUENTIAL",
        "SCENARIO-LEARN-6678-FAMILY-BLIND",
        "SCENARIO-LEARN-6678-EXACT",
        "SCENARIO-LEARN-6678-RETIREMENT",
        "SCENARIO-LEARN-6678-ISOLATION",
        "SCENARIO-LEARN-6678-RESTART",
        "SCENARIO-LEARN-6678-ROLLBACK",
        "SCENARIO-LEARN-6678-READY",
        mod.RESULT_PATH.as_posix(),
        "constraint_family_stream_ready",
    ):
        assert marker in section


def test_scenario_6678_exact_controls_cover_four_independent_families() -> None:
    """SCENARIO-LEARN-6678-EXACT: every checker accepts and rejects controls."""

    families = mod.build_family_manifest()
    controls = mod.build_exact_checker_rows(families)

    assert tuple(families) == mod.FAMILY_ORDER
    assert len({row["checker"]["function"] for row in families.values()}) == 4
    assert len(controls) == 8
    assert {(row["family"], row["control_kind"]) for row in controls} == {
        (family, kind)
        for family in mod.FAMILY_ORDER
        for kind in ("known_positive", "known_negative")
    }
    assert all(row["passed"] for row in controls)
    assert all(row["observed_exact_valid"] == row["expected_exact_valid"] for row in controls)
    assert all(row["checker_sha256"].startswith("sha256:") for row in controls)
    assert all(row["transferable_operator"] for row in families.values())


@pytest.mark.parametrize(
    ("family", "state", "reason"),
    (
        (
            "scheduling",
            {
                "jobs": [
                    {"id": "a", "start": 0, "duration": 2},
                    {"id": "b", "start": 1, "duration": 2},
                ]
            },
            "job_overlap",
        ),
        (
            "graph",
            {
                "nodes": ["a", "b"],
                "edges": [["a", "b"]],
                "colors": {"a": 1, "b": 1},
                "color_count": 2,
            },
            "edge_color_conflict",
        ),
        (
            "logic",
            {
                "variables": {"x": 1, "y": 1},
                "equations": [{"coefficients": {"x": 1, "y": 1}, "rhs": 3}],
            },
            "equation_violation",
        ),
        (
            "plan_state",
            {"steps": ["ship", "test"], "requires": {"ship": ["test"], "test": []}},
            "prerequisite_order",
        ),
    ),
)
def test_scenario_6678_checkers_return_exact_witnesses(
    family: str, state: dict[str, object], reason: str
) -> None:
    """SCENARIO-LEARN-6678-EXACT: invalid states return exact witnesses."""

    result = mod.CHECKERS[family](state)
    assert result["exact_valid"] is False
    assert result["reason"] == reason
    assert result["witness"]


def test_req_6678_checker_and_path_error_branches_are_exact() -> None:
    """REQ-LEARN-6678: malformed finite states fail with stable reasons."""

    missing = mod.check_graph(
        {"nodes": ["a", "b"], "edges": [], "colors": {"a": 1}, "color_count": 2}
    )
    out_of_range = mod.check_graph(
        {
            "nodes": ["a", "b"],
            "edges": [],
            "colors": {"a": 1, "b": 3},
            "color_count": 2,
        }
    )
    missing_step = mod.check_plan_state({"steps": ["ship"], "requires": {"ship": ["test"]}})

    assert missing["reason"] == "missing_color"
    assert out_of_range["reason"] == "color_out_of_range"
    assert missing_step["witness"] == [["test", "ship", "missing"]]
    assert mod._replace_at_path({"values": [1]}, ["values", "0"], 2) == {"values": [2]}


def test_scenario_6678_family_blind_keys_exclude_identity_and_future_data() -> None:
    """SCENARIO-LEARN-6678-FAMILY-BLIND: keys use visible typed features only."""

    events = mod.build_event_rows()
    schema = mod.build_typed_repair_schema()

    assert len(events) == mod.EVENT_COUNT == 16
    assert set(schema["excluded_fields"]) == set(mod.EXCLUDED_KEY_FIELDS)
    assert "family" not in schema["key_fields"]
    assert "event_id" not in schema["key_fields"]
    for event in events:
        material = event["applicability_key_material"]
        assert set(material) == set(mod.KEY_FIELDS)
        assert set(material).isdisjoint(mod.EXCLUDED_KEY_FIELDS)
        assert event["applicability_key"] == mod.sha256_json(material)
        encoded = mod.canonical_json(material)
        assert event["event_id"] not in encoded
        assert event["family"] not in encoded
        assert event["exact_violation_witness"]["witness_sha256"] not in encoded
        assert event["inverse_patch"]["patch_sha256"] == mod.patch_hash(event["inverse_patch"])


def test_req_6678_partitions_and_five_orders_are_sealed_and_complete() -> None:
    """REQ-LEARN-6678: partitions and five prospective orders freeze first."""

    events = mod.build_event_rows()
    families = mod.build_family_manifest(events)
    orders = mod.build_event_order_manifests(events)
    event_ids = {row["event_id"] for row in events}

    assert len(orders) == 5
    assert {row["order_id"] for row in orders} == set(mod.ORDER_SEEDS)
    for family, manifest in families.items():
        assert manifest["counts"] == {"calibration": 2, "held_family": 2, "total": 4}
        assert manifest["partitions_sha256"].startswith("sha256:")
        assert len(manifest["event_ids"]) == 4
        assert all(
            row["family"] == family for row in events if row["event_id"] in manifest["event_ids"]
        )
    for order in orders:
        assert set(order["ordered_event_ids"]) == event_ids
        assert len(order["ordered_event_ids"]) == len(event_ids)
        assert order["manifest_sha256"] == mod.order_manifest_hash(order)
        assert [row["position"] for row in order["event_rows"]] == list(range(mod.EVENT_COUNT))
        assert [row["timestamp"] for row in order["event_rows"]] == sorted(
            row["timestamp"] for row in order["event_rows"]
        )


def test_scenario_6678_prequential_admission_and_retirement_are_between_events() -> None:
    """SCENARIO-LEARN-6678-PREQUENTIAL: event t cannot read its pending patch."""

    events = mod.build_event_rows()
    order = mod.build_event_order_manifests(events)[0]
    rows = mod.build_prequential_rows(events, order)

    assert len(rows) == mod.EVENT_COUNT
    assert rows[0]["visible_commit_ids"] == []
    for index, row in enumerate(rows):
        assert row["admission_stage"] == "between_events"
        assert row["event_id"] not in row["visible_commit_ids"]
        assert row["visible_commit_ids"] == [
            prior["event_id"]
            for prior in rows[:index]
            if prior["admitted"] and not prior["retired"]
        ]
        assert row["source_repair_passed"] is True
        assert row["support_passed"] is True
        assert row["anchors_passed"] is True

    rejected = mod.admission_decision(events[0], support_ok=False)
    retired = mod.admission_decision(events[0], anchor_ok=False, was_active=True)
    assert rejected == {"admitted": False, "retired": False, "reason": "support_collapse"}
    assert retired == {"admitted": False, "retired": True, "reason": "anchor_regression"}


def test_scenario_6678_all_isolation_attacks_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6678-ISOLATION: all named mutations are detected."""

    events = mod.build_event_rows()
    orders = mod.build_event_order_manifests(events)
    rows = mod.build_isolation_attack_rows(events, orders, tmp_path)

    assert {row["attack_type"] for row in rows} == set(mod.ATTACK_TYPES)
    assert all(row["detected"] for row in rows)
    assert all(row["failed_closed"] for row in rows)
    assert all(row["reason"] for row in rows)
    assert all(row["row_sha256"] == mod.row_hash(row) for row in rows)


def test_scenarios_6678_restart_and_rollback_are_byte_exact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6678-RESTART/ROLLBACK: recovery returns exact bytes."""

    rows = mod.build_restart_rollback_rows(mod.build_event_rows(), tmp_path)
    restart = [row for row in rows if row["row_type"] == "restart"]
    rollback = [row for row in rows if row["row_type"] == "rollback"]

    assert {row["case"] for row in restart} == {
        "before_replace_old_state",
        "after_replace_new_state",
        "partial_temp_old_state",
        "corrupt_final_rejected",
    }
    assert {row["family"] for row in rollback} == set(mod.FAMILY_ORDER)
    assert all(row["passed"] for row in rows)
    assert all(row["byte_equal"] for row in rollback)
    assert all(row["recovered_class"] in {"old", "new", "rejected"} for row in restart)


def test_scenarios_6678_state_validation_and_patch_errors_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6678-ISOLATION: corrupt state and patches never coerce."""

    state = mod.empty_memory_state()
    event = mod.build_event_rows()[0]
    forward, _ = mod._memory_patch(event, state)

    assert mod.verify_memory_state({}) is False
    bad_schema = tmp_path / "bad-schema.json"
    bad_schema.write_text("{}", encoding="utf-8")
    assert mod.read_memory_state(bad_schema) == (None, "checksum_or_schema_invalid")

    corrupt_patch = deepcopy(forward)
    corrupt_patch["patch_sha256"] = "sha256:bad"
    with pytest.raises(ValueError, match="patch_checksum_corruption"):
        mod.apply_memory_patch(state, corrupt_patch)
    with pytest.raises(ValueError, match="stale_state"):
        mod.apply_memory_patch({**state, "version": 9}, forward)
    corrupt_result = deepcopy(forward)
    corrupt_result["result_state"] = {"bad": True}
    corrupt_result["result_state_sha256"] = mod.sha256_json(corrupt_result["result_state"])
    corrupt_result["patch_sha256"] = mod.patch_hash(corrupt_result)
    with pytest.raises(ValueError, match="result_state_corruption"):
        mod.apply_memory_patch(state, corrupt_result)


def test_scenario_6678_atomic_writer_cleans_temp_after_replace_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-LEARN-6678-RESTART: a failed replace leaves no partial temp."""

    def fail_replace(_source: str, _target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        mod.atomic_write_bytes(tmp_path / "state.json", b"complete")
    assert list(tmp_path.glob("*.tmp")) == []


def test_scenario_6678_ready_artifact_recomputes_from_complete_rows(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6678-READY: complete raw rows own null readiness."""

    artifact = _artifact(tmp_path, write=True)
    written = json.loads((tmp_path / mod.RESULT_PATH.name).read_text(encoding="utf-8"))

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_ready"
    assert (
        artifact["honest_verdict"]
        == "complete: exact constraint-family stream ready; no model or learning-benefit claim"
    )
    assert artifact["verdict_class"] is None
    assert artifact["constraint_family_stream_ready"] is True
    assert artifact["gate_check_summary"]["failed_check"] is None
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["aggregate_row_recomputation"]["ready"] is True
    assert artifact["reproducibility_checksum"] == mod.artifact_checksum(artifact)
    assert all(
        row["before_sha256"] == row["after_sha256"] for row in artifact["protected_files_unchanged"]
    )


def test_scenario_6678_repository_suite_is_diagnostic_not_fixture_authority(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6678-READY: unrelated repository failures stay diagnostic."""

    receipts = [
        *PASSING_TESTS,
        {
            "command": ".venv/bin/pytest tests/python -q",
            "exit_code": 3,
            "summary": "failure",
            "verification_scope": "repository_diagnostic",
            "gates_readiness": False,
        },
    ]
    artifact = mod.build_artifact(
        repo_root=REPO,
        output_path=tmp_path / "diagnostic.json",
        state_path=tmp_path / "diagnostic-state",
        date="20260827",
        duration_s=1.0,
        tests_run=receipts,
    )

    assert artifact["constraint_family_stream_ready"] is True
    assert artifact["aggregate_row_recomputation"]["checks"]["tests"] is True
    assert artifact["aggregate_row_recomputation"]["diagnostics"]["repository_suite_exit_code"] == 3


def test_req_6678_artifact_validation_fails_closed_on_tampering(tmp_path: Path) -> None:
    """REQ-LEARN-6678: row, checksum, and readiness drift cannot pass validation."""

    artifact = _artifact(tmp_path)
    tampered = deepcopy(artifact)
    tampered["isolation_attack_rows"][0]["passed"] = False
    tampered["constraint_family_stream_ready"] = True

    errors = mod.validate_artifact(tampered)
    assert "reproducibility_checksum_mismatch" in errors
    assert "attack_row_hash_mismatch" in errors
    assert "readiness_recomputation_mismatch" in errors

    assert mod.validate_artifact({})[0].startswith("missing_required_fields:")
    wrong_boundary = deepcopy(artifact)
    wrong_boundary["inference_substrate"] = "llm"
    wrong_boundary["verifier_is_oracle"] = False
    wrong_boundary["reproducibility_checksum"] = mod.artifact_checksum(wrong_boundary)
    boundary_errors = mod.validate_artifact(wrong_boundary)
    assert "inference_substrate_mismatch" in boundary_errors
    assert "oracle_boundary_mismatch" in boundary_errors


def test_req_6678_preconditions_hash_every_declared_input(tmp_path: Path) -> None:
    """REQ-LEARN-6678: source, checker, schema, roadmap, and host facts are measured."""

    rows = mod.build_preconditions(REPO, tmp_path)
    categories = {row["category"] for row in rows}
    paths = {row.get("path") for row in rows}

    assert {
        "source_corpus",
        "exact_checker",
        "repair_memory",
        "state_schema",
        "protected",
        "resource",
        "state_path",
        "substrate",
    } <= categories
    assert mod.ACTIVE_ROADMAP.as_posix() in paths
    assert mod.CONDUCTOR_PATH.as_posix() in paths
    assert all(row["available"] for row in rows)
    assert all(row.get("sha256", "sha256:measured").startswith("sha256:") for row in rows)


def test_req_6678_cli_helpers_and_atomic_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-LEARN-6678: CLI writes one validated artifact without real test recursion."""

    parsed = mod.parse_args(["--date", "20260827", "--output", str(tmp_path / "out.json")])
    assert parsed.date == "20260827"
    assert parsed.output == tmp_path / "out.json"

    monkeypatch.setattr(mod, "run_verification_commands", lambda _root: PASSING_TESTS)
    exit_code = mod.main(
        [
            "--date",
            "20260827",
            "--output",
            str(tmp_path / "cli.json"),
            "--state-path",
            str(tmp_path / "cli-state"),
        ]
    )
    assert exit_code == 0
    payload = json.loads((tmp_path / "cli.json").read_text(encoding="utf-8"))
    assert payload["constraint_family_stream_ready"] is True


def test_req_6678_command_runner_records_nonzero_exit(tmp_path: Path) -> None:
    """REQ-LEARN-6678: verification failures remain visible in test receipts."""

    row = mod.run_command(("sh", "-c", "printf failure; exit 3"), tmp_path)
    assert row["exit_code"] == 3
    assert row["summary"] == "failure"
    assert row["output_tail"] == "failure"


def test_req_6678_verification_runner_dispatches_all_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-6678: the command set is complete and ordered."""

    observed: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], _root: Path) -> dict[str, object]:
        observed.append(command)
        return {"command": " ".join(command), "exit_code": 0, "summary": "passed"}

    monkeypatch.setattr(mod, "run_command", fake_run)
    rows = mod.run_verification_commands(REPO)
    assert len(rows) == len(mod.VERIFICATION_COMMANDS)
    assert [" ".join(command) for command in observed] == list(mod.VERIFICATION_COMMANDS)
    assert all(row["gates_readiness"] for row in rows[:-1])
    assert rows[-1]["verification_scope"] == "repository_diagnostic"
    assert rows[-1]["gates_readiness"] is False
