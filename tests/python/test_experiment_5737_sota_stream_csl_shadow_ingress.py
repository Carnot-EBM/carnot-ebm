"""Tests for Exp5737 SOTA stream CSL shadow ingress.

Spec refs: REQ-LEARN-5737,
SCENARIO-LEARN-5737-CHRONOLOGICAL-INGRESS,
SCENARIO-LEARN-5737-CONTROLS,
SCENARIO-LEARN-5737-ROLLBACK,
SCENARIO-LEARN-5737-RELEASE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5737_sota_stream_csl_shadow_ingress as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5737_sota_stream_csl_shadow_ingress.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5737_sota_stream_csl_shadow_ingress.py "
    "-m pytest tests/python/test_experiment_5737_sota_stream_csl_shadow_ingress.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5737_sota_stream_csl_shadow_ingress.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5737_sota_stream_csl_shadow_ingress.json"
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
    """REQ-LEARN-5737: build the shadow ingress artifact once for schema tests."""

    base = tmp_path_factory.mktemp("exp5737")
    return mod.run(
        root=REPO,
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        ledger_path=base / mod.LEDGER_RELATIVE_PATH.name,
        test_commands=TEST_COMMANDS,
        write=True,
    )


def test_req_learn_5737_spec_declares_shadow_ingress_contract() -> None:
    """REQ-LEARN-5737: OpenSpec anchors fields, controls, and shadow-only gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5737") : spec.index("## REQ-LEARN-5640")]

    for marker in (
        "REQ-LEARN-5737",
        "SCENARIO-LEARN-5737-CHRONOLOGICAL-INGRESS",
        "SCENARIO-LEARN-5737-CONTROLS",
        "SCENARIO-LEARN-5737-ROLLBACK",
        "SCENARIO-LEARN-5737-RELEASE",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "model-proposal-label",
        "corrupted-order",
        "stale/conflict",
        "`sota_csl_ingress_ready_score` SHALL be exactly `1.0`",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_learn_5737_prefix_ingress_ledger_replays_once(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-LEARN-5737-CHRONOLOGICAL-INGRESS: prefix rows are consumed once."""

    assert mod.validate_artifact(artifact) is True
    rows = mod.load_ingress_ledger(Path(str(artifact["ingress_ledger_path"])))

    assert mod.verify_ingress_ledger(rows, artifact) is True
    assert len(rows) == artifact["prefix_row_count"] == mod.PREFIX_LENGTH
    assert [row["sequence_index"] for row in rows] == list(range(mod.PREFIX_LENGTH))
    assert len({row["row_id"] for row in rows}) == mod.PREFIX_LENGTH
    assert [row["source_previous_row_hash"] for row in rows][0] == ""
    assert all(row["ingress_row_hash"] == mod.ingress_row_hash(row) for row in rows)
    assert all(row["consumed_once"] is True for row in rows)
    assert all(row["gate_state"]["shadow_mode"] is True for row in rows)
    assert all(row["production_controller_hash_before"] == row["production_controller_hash_after"] for row in rows)
    assert all(row["state_hash_before"].startswith("sha256:") for row in rows)
    assert all(row["state_hash_after"].startswith("sha256:") for row in rows)

    for required in (
        "pre_label_decision",
        "model_proposal",
        "exact_validator_label",
        "lifecycle_operation",
        "gate_state",
        "state_hash_before",
        "state_hash_after",
    ):
        assert required in rows[0]


def test_scenario_learn_5737_controls_and_strata_pass(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-LEARN-5737-CONTROLS: exact labels beat diagnostic controls."""

    exact = artifact["exact_label_update_results"]
    no_update = artifact["no_update_control_results"]
    proposal = artifact["proposal_label_control_results"]
    corrupted = artifact["corrupted_order_results"]
    stale_conflict = artifact["stale_conflict_control_results"]

    assert artifact["session_count"] >= 30
    assert artifact["suffix_improvement"] > 0.0
    assert exact["suffix_accuracy"] > no_update["suffix_accuracy"]
    assert exact["suffix_accuracy"] > proposal["suffix_accuracy"]
    assert exact["suffix_accuracy"] > corrupted["suffix_accuracy"]
    assert proposal["diagnostic_only"] is True
    assert proposal["protected_controller_mutated"] is False
    assert stale_conflict["stale_rejected"] is True
    assert stale_conflict["conflict_rejected"] is True
    assert stale_conflict["accepted_update_count"] == 0
    assert all(row["exact_accuracy"] >= row["proposal_accuracy"] for row in artifact["family_model_strata"])
    assert artifact["first_changed_decisions"]["exact_label_updates"]["row_id"]
    assert set(artifact["arm_configs"]) == {
        mod.EXACT_LABEL_ARM,
        mod.NO_UPDATE_ARM,
        mod.MODEL_PROPOSAL_ARM,
        mod.CORRUPTED_ORDER_ARM,
        mod.STALE_CONFLICT_ARM,
    }


def test_scenario_learn_5737_release_fields_hashes_and_ready_score(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-LEARN-5737-RELEASE: release gates pass only for shadow-safe ingress."""

    assert artifact["preconditions_checked"]["all_passed"] is True
    assert artifact["upstream_gate_receipts"]["all_passed"] is True
    assert artifact["stream_root_commitment"].startswith("sha256:")
    assert artifact["prefix_hash"].startswith("sha256:")
    assert artifact["suffix_hash"].startswith("sha256:")
    assert artifact["lifecycle_hash"].startswith("sha256:")
    assert artifact["validator_hashes"]["all_validated"] is True
    assert artifact["prefix_retention_delta"] <= 0.0
    assert artifact["unsafe_update_count"] == 0
    assert artifact["rollback_state_hash_matches"] is True
    assert artifact["model_weight_mutation"] is False
    assert artifact["production_default_enabled"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["sota_csl_ingress_ready_score"] == 1.0
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["test_commands"] == TEST_COMMANDS

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert artifact["field_principles"][field] == mod.REQUIRED_FIELD_PRINCIPLES[field]
    for field in artifact:
        assert field in artifact["field_principles"]


def test_req_learn_5737_run_writes_stable_artifact(
    artifact: dict[str, object],
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5737: run output, ledger, and checksum replay exactly."""

    destination = tmp_path / mod.RESULT_RELATIVE_PATH.name
    ledger = tmp_path / mod.LEDGER_RELATIVE_PATH.name
    written = mod.run(
        root=REPO,
        result_path=destination,
        ledger_path=ledger,
        test_commands=TEST_COMMANDS,
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert loaded == written
    assert mod.validate_artifact(written) is True
    assert Path(str(written["ingress_ledger_path"])) == ledger
    assert mod.verify_ingress_ledger(mod.load_ingress_ledger(ledger), written) is True
    assert written["reproducibility_checksum"] == mod.reproducibility_checksum(written)
    assert written["stream_root_commitment"] == artifact["stream_root_commitment"]


def test_req_learn_5737_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5737: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.run(
        root=REPO,
        result_path=RESULT_PATH,
        ledger_path=result["ingress_ledger_path"],
        test_commands=result["test_commands"],
        write=False,
    )

    assert result == replay
    assert result["sota_csl_ingress_ready_score"] == 1.0
    assert result["honest_verdict"].startswith("complete:")
    assert result["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    mod.validate_artifact(result)


def test_req_learn_5737_validation_fails_closed(artifact: dict[str, object]) -> None:
    """REQ-LEARN-5737: artifact validation rejects unsafe or stale readiness claims."""

    cases: list[tuple[str, dict[str, object]]] = []
    for field, value, expected in (
        ("suffix_improvement", 0.0, "suffix_improvement"),
        ("prefix_retention_delta", 0.1, "prefix_retention_delta"),
        ("unsafe_update_count", 1, "unsafe_update_count"),
        ("rollback_state_hash_matches", False, "rollback_state_hash_matches"),
        ("model_weight_mutation", True, "model_weight_mutation"),
        ("production_default_enabled", True, "production_default_enabled"),
    ):
        bad = deepcopy(artifact)
        bad[field] = value
        cases.append((expected, bad))

    bad = deepcopy(artifact)
    bad["preconditions_checked"]["all_passed"] = False
    cases.append(("preconditions_checked", bad))

    bad = deepcopy(artifact)
    bad["upstream_gate_receipts"]["all_passed"] = False
    cases.append(("upstream_gate_receipts", bad))

    bad = deepcopy(artifact)
    bad["proposal_label_control_results"]["protected_controller_mutated"] = True
    cases.append(("proposal_label_control_results", bad))

    bad = deepcopy(artifact)
    bad["corrupted_order_results"]["exact_arm_outperformed"] = False
    cases.append(("corrupted_order_results", bad))

    bad = deepcopy(artifact)
    bad["field_principles"].pop("suffix_improvement")
    cases.append(("field_principles", bad))

    bad = deepcopy(artifact)
    bad.pop("suffix_improvement")
    cases.append(("missing required fields", bad))

    bad = deepcopy(artifact)
    bad["sota_csl_ingress_ready_score"] = 0.0
    cases.append(("sota_csl_ingress_ready_score", bad))

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "complete: stale"
    cases.append(("honest_verdict", bad))

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    cases.append(("reproducibility_checksum", bad))

    for expected, bad_artifact in cases:
        if expected not in {"honest_verdict", "reproducibility_checksum", "sota_csl_ingress_ready_score"}:
            bad_artifact["sota_csl_ingress_ready_score"] = mod.sota_csl_ingress_ready_score(
                bad_artifact
            )
            bad_artifact["honest_verdict"] = mod.honest_verdict(bad_artifact)
            bad_artifact["reproducibility_checksum"] = mod.reproducibility_checksum(
                bad_artifact
            )
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad_artifact)


def test_req_learn_5737_helper_edges(artifact: dict[str, object]) -> None:
    """REQ-LEARN-5737: helper edge cases remain deterministic and auditable."""

    rows = mod.load_ingress_ledger(Path(str(artifact["ingress_ledger_path"])))
    bad_hash = deepcopy(rows)
    bad_hash[0]["ingress_row_hash"] = "sha256:bad"

    assert mod.ingress_row_hash(rows[0]) == rows[0]["ingress_row_hash"]
    assert mod.verify_ingress_ledger(bad_hash, artifact) is False
    assert mod.verify_ingress_ledger(rows[:-1], artifact) is False

    bad_sequence = deepcopy(rows)
    bad_sequence[1]["sequence_index"] = 99
    bad_sequence[1]["ingress_row_hash"] = mod.ingress_row_hash(bad_sequence[1])
    assert mod.verify_ingress_ledger(bad_sequence, artifact) is False

    bad_chain = deepcopy(rows)
    bad_chain[1]["previous_ingress_row_hash"] = "sha256:bad"
    bad_chain[1]["ingress_row_hash"] = mod.ingress_row_hash(bad_chain[1])
    assert mod.verify_ingress_ledger(bad_chain, artifact) is False

    duplicate = deepcopy(rows)
    duplicate[1]["row_id"] = duplicate[0]["row_id"]
    duplicate[1]["ingress_row_hash"] = mod.ingress_row_hash(duplicate[1])
    assert mod.verify_ingress_ledger(duplicate, artifact) is False

    suffix = mod.exp5734.read_row_manifest(REPO / mod.EXP5734_ROW_MANIFEST_RELATIVE_PATH)[
        mod.PREFIX_LENGTH :
    ]
    proposals = [row["selected_label"] for row in suffix]
    assert (
        mod._first_changed_decision(
            before_predictions=proposals,
            after_predictions=proposals,
            rows=suffix,
            arm=mod.NO_UPDATE_ARM,
        )
        is None
    )
    assert mod.artifact_errors({}) == ["missing required fields: " + str(list(mod.REQUIRED_ARTIFACT_FIELDS))]
    assert mod.sota_csl_ingress_ready_score({}) == 0.0
    assert mod.honest_verdict({}).startswith("blocked:")

    non_mapping_principles = deepcopy(artifact)
    non_mapping_principles["field_principles"] = []
    non_mapping_principles["sota_csl_ingress_ready_score"] = mod.sota_csl_ingress_ready_score(
        non_mapping_principles
    )
    non_mapping_principles["honest_verdict"] = mod.honest_verdict(non_mapping_principles)
    non_mapping_principles["reproducibility_checksum"] = mod.reproducibility_checksum(
        non_mapping_principles
    )
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(non_mapping_principles)
