"""Tests for Exp5736 CSL lifecycle conflict rollback.

Spec refs: REQ-LEARN-5736,
SCENARIO-LEARN-5736-LIFECYCLE,
SCENARIO-LEARN-5736-CONFLICT,
SCENARIO-LEARN-5736-ROLLBACK,
SCENARIO-LEARN-5736-RELEASE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5736_csl_lifecycle_conflict_rollback as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5736_csl_lifecycle_conflict_rollback.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5736_csl_lifecycle_conflict_rollback.py "
    "-m pytest tests/python/test_experiment_5736_csl_lifecycle_conflict_rollback.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5736_csl_lifecycle_conflict_rollback.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5736_csl_lifecycle_conflict_rollback.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """REQ-LEARN-5736: build the lifecycle artifact once for schema tests."""

    base = tmp_path_factory.mktemp("exp5736")
    return mod.run(
        root=REPO,
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        ledger_path=base / mod.LEDGER_RELATIVE_PATH.name,
        checkpoint_dir=base / "checkpoints",
        test_commands=TEST_COMMANDS,
        write=True,
    )


def test_req_learn_5736_spec_declares_lifecycle_contract() -> None:
    """REQ-LEARN-5736: OpenSpec anchors operations, failures, and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5736") : spec.index("## REQ-LEARN-5640")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5736",
        "SCENARIO-LEARN-5736-LIFECYCLE",
        "SCENARIO-LEARN-5736-CONFLICT",
        "SCENARIO-LEARN-5736-ROLLBACK",
        "SCENARIO-LEARN-5736-RELEASE",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "`remember`",
        "`update`",
        "`supersede`",
        "`forget`",
        "conflict `reject`",
        "`rollback`",
        "`recover`",
        "crash before write",
        "corrupted checkpoint",
        "duplicate event ID",
        "replayed stale advice",
    ):
        assert marker in section

    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_5736_lifecycle_schema_and_ledger_replay(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-LEARN-5736-LIFECYCLE: typed rows replay deterministically."""

    assert mod.validate_artifact(artifact) is True
    rows = mod.load_operation_ledger(Path(str(artifact["operation_ledger_path"])))
    schema_fields = set(artifact["transition_schema"]["required_fields"])

    assert mod.verify_operation_ledger(rows, artifact) is True
    assert len(rows) == artifact["operation_counts"]["total"]
    assert mod.replay_operation_ledger(rows, artifact)["passed"] is True
    assert set(mod.TRANSITION_SCHEMA_REQUIRED_FIELDS) <= schema_fields
    assert all(set(mod.TRANSITION_SCHEMA_REQUIRED_FIELDS) <= set(row) for row in rows)
    assert all(row["transition_hash"] == mod.transition_row_hash(row) for row in rows)
    assert all(row["exact_validator_receipt"]["receipt_hash"].startswith("sha256:") for row in rows)
    assert all(row["predecessor_hash"]["combined_hash"].startswith("sha256:") for row in rows)
    assert all(row["successor_hash"]["combined_hash"].startswith("sha256:") for row in rows)

    for operation in mod.LIFECYCLE_OPERATIONS:
        assert artifact["operation_counts"]["by_operation"][operation] > 0
    assert artifact["ledger_replay_equivalence"]["passed"] is True
    assert artifact["ledger_replay_equivalence"]["all_valid_operations_replayed"] is True
    assert artifact["ledger_replay_equivalence"]["all_invalid_operations_rejected"] is True


def test_scenario_learn_5736_conflicts_reject_with_zero_propagation(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-LEARN-5736-CONFLICT: invalid advice fails closed."""

    rows = mod.load_operation_ledger(Path(str(artifact["operation_ledger_path"])))
    rejected = [row for row in rows if row["accepted"] is False]

    assert artifact["rejected_transition_count"] == len(rejected)
    assert artifact["rejected_transition_count"] >= 6
    assert artifact["unsafe_propagation_count"] == 0
    assert all(row["successor_hash"] == row["predecessor_hash"] for row in rejected)
    assert all(row["propagation_depth"] == 0 for row in rejected)
    assert all(row["protected_prefix_effect"]["passed"] is True for row in rejected)
    assert all(row["first_changed_decision"] is None for row in rejected)

    case_kinds = {case["case_kind"] for case in artifact["conflict_cases"]}
    assert case_kinds >= {
        "stale",
        "contradictory",
        "superseded",
        "forgotten",
        "reordered",
        "duplicate_event_id",
        "replayed_stale_advice",
    }
    assert all(case["rejected"] is True for case in artifact["conflict_cases"])
    assert artifact["corruption_controls"]["duplicate_event_ids"]["rejected"] is True
    assert artifact["corruption_controls"]["replayed_stale_advice"]["rejected"] is True


def test_scenario_learn_5736_crash_corruption_rollback_and_recover(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-LEARN-5736-ROLLBACK: recovery restores exact hashes."""

    crash_points = {row["injection_point"] for row in artifact["crash_injection_matrix"]}
    assert crash_points == set(mod.CRASH_INJECTION_POINTS)
    assert all(row["fail_closed"] is True for row in artifact["crash_injection_matrix"])
    assert all(row["recovered"] is True for row in artifact["crash_injection_matrix"])
    assert all(row["recovery_latency_ms"] <= mod.MAX_RECOVERY_LATENCY_MS for row in artifact["crash_injection_matrix"])

    corruption = artifact["corruption_controls"]
    for control in (
        "corrupted_checkpoints",
        "orphan_ledger_entries",
        "duplicate_event_ids",
        "replayed_stale_advice",
    ):
        assert corruption[control]["detected"] is True
        assert corruption[control]["rejected"] is True

    rollback_receipts = [
        row for row in artifact["recovery_receipts"] if row["recovery_type"] == "rollback"
    ]
    recover_receipts = [
        row for row in artifact["recovery_receipts"] if row["recovery_type"] == "recover"
    ]
    assert rollback_receipts
    assert recover_receipts
    assert artifact["rollback_state_hash_matches"] is True
    assert all(row["exact_hash_match"] is True for row in rollback_receipts)
    assert all(row["exact_hash_match"] is True for row in recover_receipts)


def test_scenario_learn_5736_release_fields_retention_and_ready_score(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-LEARN-5736-RELEASE: release gates pass only when safe."""

    assert artifact["preconditions_checked"]["all_passed"] is True
    assert artifact["upstream_gate_receipts"]["all_passed"] is True
    assert artifact["upstream_hash"] == mod.EXPECTED_EXP5735_HASH
    assert artifact["suffix_commitment"]["session_count"] == mod.SESSION_COUNT
    assert len(artifact["random_seeds"]) == mod.SESSION_COUNT
    assert artifact["epsilon"] == pytest.approx(mod.EPSILON)
    assert artifact["delta"] == pytest.approx(mod.DELTA)
    assert artifact["prefix_retention_delta"] <= mod.PREFIX_RETENTION_MARGIN
    assert artifact["suffix_improvement"] > 0.0
    assert artifact["statistical_model_check_receipt"]["passes"] is True
    assert artifact["model_weight_mutation"] is False
    assert artifact["production_default_enabled"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["csl_lifecycle_ready_score"] == 1.0
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["test_commands"] == TEST_COMMANDS

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert artifact["field_principles"][field] == mod.REQUIRED_FIELD_PRINCIPLES[field]
    for field in artifact:
        assert field in artifact["field_principles"]


def test_req_learn_5736_run_writes_stable_artifact(
    artifact: dict[str, object],
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5736: run output, ledger, and checksum replay exactly."""

    destination = tmp_path / mod.RESULT_RELATIVE_PATH.name
    ledger = tmp_path / mod.LEDGER_RELATIVE_PATH.name
    written = mod.run(
        root=REPO,
        result_path=destination,
        ledger_path=ledger,
        checkpoint_dir=tmp_path / "checkpoints",
        test_commands=TEST_COMMANDS,
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert loaded == written
    assert mod.validate_artifact(written) is True
    assert Path(str(written["operation_ledger_path"])) == ledger
    assert mod.verify_operation_ledger(mod.load_operation_ledger(ledger), written) is True
    assert written["reproducibility_checksum"] == mod.reproducibility_checksum(written)
    assert written["suffix_commitment"]["suffix_order_hash"] == artifact["suffix_commitment"]["suffix_order_hash"]


def test_req_learn_5736_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5736: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.run(
        root=REPO,
        result_path=RESULT_PATH,
        ledger_path=result["operation_ledger_path"],
        checkpoint_dir=REPO / mod.CHECKPOINT_RELATIVE_DIR,
        test_commands=result["test_commands"],
        write=False,
    )

    assert result == replay
    assert result["csl_lifecycle_ready_score"] == 1.0
    assert result["honest_verdict"].startswith("complete:")
    assert result["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    mod.validate_artifact(result)


def test_req_learn_5736_validation_fails_closed(artifact: dict[str, object]) -> None:
    """REQ-LEARN-5736: artifact validation rejects unsafe drift."""

    cases: list[tuple[str, dict[str, object]]] = []
    for field, value, expected in (
        ("suffix_improvement", 0.0, "suffix_improvement"),
        ("prefix_retention_delta", mod.PREFIX_RETENTION_MARGIN + 0.1, "prefix_retention_delta"),
        ("unsafe_propagation_count", 1, "unsafe_propagation_count"),
        ("rollback_state_hash_matches", False, "rollback_state_hash_matches"),
        ("model_weight_mutation", True, "model_weight_mutation"),
        ("production_default_enabled", True, "production_default_enabled"),
    ):
        bad = deepcopy(artifact)
        bad[field] = value
        cases.append((expected, bad))

    bad = deepcopy(artifact)
    bad["ledger_replay_equivalence"]["passed"] = False
    cases.append(("ledger_replay_equivalence", bad))

    bad = deepcopy(artifact)
    bad["statistical_model_check_receipt"]["passes"] = False
    cases.append(("statistical_model_check_receipt", bad))

    bad = deepcopy(artifact)
    bad["field_principles"].pop("suffix_improvement")
    cases.append(("field_principles", bad))

    bad = deepcopy(artifact)
    bad.pop("suffix_improvement")
    cases.append(("missing required fields", bad))

    bad = deepcopy(artifact)
    bad["csl_lifecycle_ready_score"] = 0.0
    cases.append(("csl_lifecycle_ready_score", bad))

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "complete: stale"
    cases.append(("honest_verdict", bad))

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    cases.append(("reproducibility_checksum", bad))

    for expected, bad_artifact in cases:
        if expected not in {"honest_verdict", "reproducibility_checksum", "csl_lifecycle_ready_score"}:
            bad_artifact["csl_lifecycle_ready_score"] = mod.csl_lifecycle_ready_score(bad_artifact)
            bad_artifact["honest_verdict"] = mod.honest_verdict(bad_artifact)
            bad_artifact["reproducibility_checksum"] = mod.reproducibility_checksum(bad_artifact)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad_artifact)


def test_req_learn_5736_helper_edges(artifact: dict[str, object]) -> None:
    """REQ-LEARN-5736: helper edge cases remain deterministic and auditable."""

    rows = mod.load_operation_ledger(Path(str(artifact["operation_ledger_path"])))
    first = rows[0]
    bad_hash = deepcopy(first)
    bad_hash["transition_hash"] = "sha256:bad"

    assert mod.transition_row_hash(first) == first["transition_hash"]
    assert mod.verify_operation_ledger([bad_hash, *rows[1:]], artifact) is False
    assert mod.replay_operation_ledger([bad_hash, *rows[1:]], artifact)["reason"] == "transition_hash"
    assert mod.csl_lifecycle_ready_score({}) == 0.0
    assert mod.honest_verdict({}).startswith("blocked:")
    assert mod.replay_operation_ledger([], artifact)["passed"] is False
    assert mod.lifecycle_operation_counts(rows)["total"] == len(rows)
    assert mod.unsafe_propagation_count(rows) == 0
    assert mod.verify_checkpoint_payloads(artifact["recovery_receipts"]) is True

    bad_predecessor = deepcopy(rows)
    bad_predecessor[1]["predecessor_hash"] = rows[0]["predecessor_hash"]
    bad_predecessor[1]["transition_hash"] = mod.transition_row_hash(bad_predecessor[1])
    assert mod.replay_operation_ledger(bad_predecessor, artifact)["reason"] == "predecessor_hash"

    rejected_index = next(index for index, row in enumerate(rows) if row["accepted"] is False)
    bad_rejected = deepcopy(rows)
    bad_rejected[rejected_index]["successor_hash"] = rows[0]["predecessor_hash"]
    bad_rejected[rejected_index]["transition_hash"] = mod.transition_row_hash(
        bad_rejected[rejected_index]
    )
    assert mod.replay_operation_ledger(bad_rejected, artifact)["reason"] == "rejected_mutated"

    bad_checkpoint_hash = deepcopy(artifact["recovery_receipts"])
    checkpoint_receipt = next(row for row in bad_checkpoint_hash if "checkpoint_path" in row)
    checkpoint_receipt["checkpoint_hash"] = "sha256:bad"
    assert mod.verify_checkpoint_payloads(bad_checkpoint_hash) is False

    bad_embedded_hash = deepcopy(artifact["recovery_receipts"])
    checkpoint_receipt = next(row for row in bad_embedded_hash if "checkpoint_path" in row)
    checkpoint_receipt["embedded_hash"] = {"state_hash": "sha256:bad"}
    assert mod.verify_checkpoint_payloads(bad_embedded_hash) is False

    context = mod._load_context(REPO)
    assert mod._row_by_id(context, "missing-row") is None
    system = mod._initial_system()
    before_hash = mod._system_hash(system)
    row = context.suffix_rows[0]
    wrong_label_event = mod._event(
        event_id="edge-wrong-label",
        operation="remember",
        trigger="edge_exact_reject",
        target="edge/constraint",
        scope=row.stream_id,
        row=row,
        proposed_label=-int(row.label),
        value={"label": -int(row.label), "row_id": row.row_id},
        case_kind="contradictory",
        claimed_predecessor_hash=before_hash["combined_hash"],
    )
    wrong_receipt = mod._exact_validator_receipt(wrong_label_event, row)
    assert (
        mod._rejection_reason(
            event=wrong_label_event,
            exact_receipt=wrong_receipt,
            system=system,
            attempted_event_ids=set(),
            before_hash=before_hash,
        )
        == "exact_validator_reject"
    )

    system.entries["edge/constraint"] = {"status": "active"}
    active_event = deepcopy(wrong_label_event)
    active_event["evidence"]["proposed_label"] = row.label
    active_event["evidence"]["value"]["label"] = row.label
    active_receipt = mod._exact_validator_receipt(active_event, row)
    assert (
        mod._rejection_reason(
            event=active_event,
            exact_receipt=active_receipt,
            system=system,
            attempted_event_ids=set(),
            before_hash=before_hash,
        )
        == "target_already_active"
    )

    non_mapping_principles = deepcopy(artifact)
    non_mapping_principles["field_principles"] = []
    non_mapping_principles["csl_lifecycle_ready_score"] = mod.csl_lifecycle_ready_score(
        non_mapping_principles
    )
    non_mapping_principles["honest_verdict"] = mod.honest_verdict(non_mapping_principles)
    non_mapping_principles["reproducibility_checksum"] = mod.reproducibility_checksum(
        non_mapping_principles
    )
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(non_mapping_principles)
