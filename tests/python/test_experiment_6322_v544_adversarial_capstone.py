"""Tests for Exp6322 V544 adversarial capstone.

Spec refs: REQ-INFRA-6322, SCENARIO-INFRA-6322-1,
SCENARIO-INFRA-6322-2, SCENARIO-INFRA-6322-3,
SCENARIO-INFRA-6322-4, SCENARIO-INFRA-6322-5,
SCENARIO-INFRA-6322-6.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6322_v544_adversarial_capstone as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-harnesses/spec.md"


def test_req_infra_6322_spec_declares_required_contract() -> None:
    """REQ-INFRA-6322: OpenSpec names fields and V544 capstone scenarios."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6322") :]

    for marker in (
        "SCENARIO-INFRA-6322-1",
        "SCENARIO-INFRA-6322-2",
        "SCENARIO-INFRA-6322-3",
        "SCENARIO-INFRA-6322-4",
        "SCENARIO-INFRA-6322-5",
        "SCENARIO-INFRA-6322-6",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "shared_bus_promotion_allowed",
        "arc_solve_claim_allowed",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenarios_infra_6322_counts_and_exact_matrix_preserve_states() -> None:
    """SCENARIO-INFRA-6322-1 and SCENARIO-INFRA-6322-2: exact states stay exact."""

    report = mod.build_report(
        REPO,
        date="20260812",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )

    assert mod.validate_report(report) == []
    assert len(report["declared_task_ids_and_deliverables"]) == 13
    assert len(report["task_terminal_matrix"]) == 13

    matrix = report["task_terminal_matrix"]
    assert (
        matrix["exp6312-model-local-representation-surface-preflight"]["terminal_class"] == "null"
    )
    assert matrix["exp6314-three-family-model-local-state-corpus"]["terminal_class"] == "skipped"
    assert matrix["exp6314-three-family-model-local-state-corpus"]["raw_blocked_status"] is True
    assert (
        matrix["exp6315-model-local-paired-difference-energy-probes"]["terminal_class"] == "missing"
    )
    assert matrix["exp6316-model-local-probe-integrity-audit"]["terminal_class"] == "flagged"
    assert (
        matrix["exp6317-live-three-family-model-local-verifier-benchmark"]["terminal_class"]
        == "missing"
    )
    assert matrix["exp6321-arc-target-licensed-route-live-shadow-ab"]["shadow_only"] is True

    counts = report[
        "missing_nonterminal_flagged_null_blocked_skipped_oracle_only_safety_only_shadow_only_ready_and_positive_counts"
    ]
    assert counts["task_count"] == 13
    assert counts["terminal_class_task_count_sum"] == 13
    assert counts["missing"] == 2
    assert counts["nonterminal"] == 2
    assert counts["flagged"] == 1
    assert counts["null"] == 2
    assert counts["blocked"] == 1
    assert counts["skipped"] == 1
    assert counts["oracle_only"] >= 1
    assert counts["safety_only"] == 1
    assert counts["shadow_only"] == 1
    assert counts["ready"] >= 1
    assert counts["positive"] == 3


def test_scenarios_infra_6322_branch_gates_and_laundering_guards() -> None:
    """SCENARIO-INFRA-6322-3 and SCENARIO-INFRA-6322-4: branches stay independent."""

    report = mod.build_report(
        REPO,
        date="20260812",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )

    for field in mod.LAUNDERING_GUARD_FIELDS:
        assert report[field] is False

    matrix = report["branch_promotion_matrix"]
    assert matrix["model_local_verification"]["promotion_allowed"] is False
    assert matrix["model_local_verification"]["terminal_state"] == "closed_null_or_flagged"
    assert (
        "results/experiment_6315_model_local_paired_difference_energy_probes.json"
        in matrix["model_local_verification"]["failed_or_underpowered_cells"]
    )
    assert matrix["versioned_factor_local_learning"]["promotion_allowed"] is True
    assert matrix["feedback_directed_search"]["promotion_allowed"] is True
    assert matrix["online_self_evolution_safety"]["promotion_allowed"] is True
    assert matrix["online_self_evolution_safety"]["safety_only"] is True
    assert matrix["arc_live_shadow"]["promotion_allowed"] is True
    assert matrix["arc_live_shadow"]["solve_claim_allowed"] is False

    assert report["model_local_representation_verdict"]["ready_score"] == 0.0
    assert report["model_local_probe_integrity_verdict"]["promotion_allowed"] is False
    assert report["live_model_local_verifier_verdict"]["terminal_class"] == "missing"
    assert report["versioned_factor_local_learning_verdict"]["cross_family_transfer_count"] == 0
    assert report["feedback_directed_search_verdict"]["protected_validation_reuse_count"] == 0
    assert report["online_self_evolution_safety_verdict"]["utility_claim_allowed"] is False
    assert report["arc_live_shadow_verdict"]["levels_credited"] == 0


def test_scenarios_infra_6322_retirements_schema_and_write(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6322-5 and SCENARIO-INFRA-6322-6: receipts and schema are exact."""

    report = mod.build_report(
        REPO,
        date="20260812",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(report["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(report["field_provenance"])
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    assert report["honest_verdict"].startswith("complete:")
    assert report["failed_experiment_rerun_retirements"]["rule_fired_count"] == 0
    assert report["exclusion_manifest_updates"]["updated"] is False
    assert (
        report["protected_files_unchanged"]["paths"]["research-roadmap.yaml"]["unchanged"] is True
    )

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    path = mod.write_report(report, REPO, env={ARTIFACT_ROOT_ENV: str(artifact_root)})

    assert path == artifact_root / mod.RESULT_RELATIVE_PATH.name
    assert json.loads(path.read_text(encoding="utf-8")) == report

    same = mod.failed_experiment_rerun_retirements(
        [
            {
                "id": "same",
                "prior_failures": [
                    {
                        "experiment_id": "exp1",
                        "verdict": "complete: same",
                        "retire_if_same_verdict": True,
                    }
                ],
            }
        ],
        {"same": {"honest_verdict_raw": "complete: same", "terminal_class": "complete"}},
    )
    assert same["rule_fired_count"] == 1
    assert same["actions"][0]["action"] == "retire_if_same_verdict_rule_fired"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shared_bus_promotion_allowed", True, "shared_bus_promotion_allowed"),
        (
            "cross_family_transfer_promotion_allowed",
            True,
            "cross_family_transfer_promotion_allowed",
        ),
        (
            "exact_oracle_as_learned_verifier_allowed",
            True,
            "exact_oracle_as_learned_verifier_allowed",
        ),
        (
            "protected_validation_as_progress_allowed",
            True,
            "protected_validation_as_progress_allowed",
        ),
        ("arc_solve_claim_allowed", True, "arc_solve_claim_allowed"),
        ("field_principles", {}, "missing field_principles entry"),
        ("field_provenance", {}, "missing field_provenance entry"),
        ("honest_verdict", "not_terminal", "honest_verdict lacks terminal prefix"),
    ],
)
def test_validate_report_rejects_claim_laundering_and_schema_breaks(
    field: str, value: object, message: str
) -> None:
    """REQ-INFRA-6322: validation rejects malformed reports."""

    report = mod.build_report(
        REPO,
        date="20260812",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )
    bad = copy.deepcopy(report)
    bad[field] = value
    if field not in {"reproducibility_checksum", "duration_s"}:
        bad["reproducibility_checksum"] = mod.payload_checksum(bad)

    errors = mod.validate_report(bad)
    assert any(message in error for error in errors)


def test_helper_edge_paths_for_req_infra_6322(tmp_path: Path) -> None:
    """REQ-INFRA-6322: helper readers and bare-value handling fail closed."""

    assert mod.read_yaml_mapping(tmp_path / "missing.yaml") == {}
    assert mod.roadmap_tasks({"tasks": "not-a-list"}) == []

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    _, bad_meta = mod.read_json_mapping(bad_json)
    assert bad_meta["loadable"] is False
    assert str(bad_meta["error"]).startswith("json_error:")

    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[1, 2, 3]", encoding="utf-8")
    _, scalar_meta = mod.read_json_mapping(scalar_json)
    assert scalar_meta["error"] == "json_not_mapping"

    assert mod._bare_value({"x": {"value": 3, "principle": "wrapped"}}, "x") == 3
    skipped = mod.failed_experiment_rerun_retirements(
        [
            {
                "id": "skip",
                "prior_failures": [
                    "not-a-map",
                    {"verdict": "complete: same", "retire_if_same_verdict": False},
                ],
            }
        ],
        {"skip": {"honest_verdict_raw": "complete: same", "terminal_class": "complete"}},
    )
    assert skipped["actions"] == []
    assert mod._status_from_commands([{"command": "bad", "exit_code": 2}])[0] == "blocked"


def test_external_receipts_and_cli_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-INFRA-6322-6: external receipts and CLI dispatch are deterministic."""

    receipt_path = tmp_path / "receipts.json"
    monkeypatch.setattr(mod, "EXTERNAL_TEST_RECEIPT_PATH", receipt_path)
    assert mod._read_external_test_receipts() == [{"command": mod.RUN_COMMAND, "exit_code": 0}]

    receipt_path.write_text("{bad", encoding="utf-8")
    assert mod._read_external_test_receipts() == [{"command": mod.RUN_COMMAND, "exit_code": 0}]

    receipt_path.write_text(json.dumps({"focused": 0, "lint": 1}), encoding="utf-8")
    assert mod._read_external_test_receipts() == [
        {"command": "focused", "exit_code": 0},
        {"command": "lint", "exit_code": 1},
    ]

    receipt_path.write_text(
        json.dumps([{"command": "coverage", "exit_code": 0}, {"bad": 1}]),
        encoding="utf-8",
    )
    assert mod._read_external_test_receipts() == [{"command": "coverage", "exit_code": 0}]

    receipt_path.write_text("[]", encoding="utf-8")
    assert mod._read_external_test_receipts() == [{"command": mod.RUN_COMMAND, "exit_code": 0}]

    monkeypatch.setattr(mod, "run", lambda *, date: {"status": f"complete-{date}"})
    assert mod.main(["--date", "20260812"]) == 0
    assert "experiment_6322_v544_adversarial_capstone.json" in capsys.readouterr().out


def test_validate_report_edge_failures_and_retro_run_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-INFRA-6322-6: validation and run/write wrappers cover edge cases."""

    base = mod.build_report(
        REPO,
        date="20260812",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )

    invalid = {"status": "complete"}
    invalid_errors = mod.validate_report(invalid)
    assert "missing required field: roadmap_path_and_hash" in invalid_errors
    assert "field_principles is not a mapping" in invalid_errors
    assert "field_provenance is not a mapping" in invalid_errors
    assert "counts field is not a mapping" in invalid_errors
    assert "reproducibility_checksum missing" in invalid_errors

    blocked = mod.build_report(
        REPO,
        date="20260812",
        command_receipts=[{"command": mod.FULL_PYTEST_COMMAND, "exit_code": 2}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("complete:")
    assert mod.validate_report(blocked) == []

    bad_counts = copy.deepcopy(base)
    counts_key = "missing_nonterminal_flagged_null_blocked_skipped_oracle_only_safety_only_shadow_only_ready_and_positive_counts"
    bad_counts[counts_key]["task_count"] = 12
    bad_counts[counts_key]["terminal_class_task_count_sum"] = 12
    bad_counts[counts_key]["count_principles"] = {}
    bad_counts["reproducibility_checksum"] = mod.payload_checksum(bad_counts)
    count_errors = mod.validate_report(bad_counts)
    assert "task_count must be 13" in count_errors
    assert "terminal class counts must conserve 13 tasks" in count_errors
    assert "missing count principle: missing" in count_errors

    bad_arc = copy.deepcopy(base)
    bad_arc["arc_live_shadow_verdict"]["solve_claimed"] = True
    bad_arc["arc_live_shadow_verdict"]["levels_credited"] = 1
    bad_arc["arc_live_shadow_verdict"]["registry_update_count"] = 1
    bad_arc["verifier_is_oracle"] = True
    bad_arc["reproducibility_checksum"] = mod.payload_checksum(bad_arc)
    arc_errors = mod.validate_report(bad_arc)
    assert "arc solve_claimed must be false" in arc_errors
    assert "arc levels_credited must be zero" in arc_errors
    assert "arc registry_update_count must be zero" in arc_errors
    assert "verifier_is_oracle must be false" in arc_errors

    bad_arc_type = copy.deepcopy(base)
    bad_arc_type["arc_live_shadow_verdict"] = "bad"
    bad_arc_type["reproducibility_checksum"] = mod.payload_checksum(bad_arc_type)
    assert "arc_live_shadow_verdict is not a mapping" in mod.validate_report(bad_arc_type)

    bad_checksum = copy.deepcopy(base)
    bad_checksum["status"] = "changed"
    assert "reproducibility_checksum mismatch" in mod.validate_report(bad_checksum)

    with pytest.raises(ValueError, match="invalid Exp6322 report"):
        mod.write_report({"status": "complete"}, REPO)

    retro_root = tmp_path / "retro-root"
    (retro_root / "ops").mkdir(parents=True)
    retro = mod.write_operational_retro(
        retro_root, {"run_date": "20260812", "branch_promotion_matrix": {"x": True}}
    )
    assert retro["present"] is True
    assert retro["sha256"]

    monkeypatch.setattr(
        mod,
        "_read_external_test_receipts",
        lambda: [{"command": "focused", "exit_code": 0}],
    )
    monkeypatch.setattr(mod, "git_status_lines", lambda root: [" M fixture"])
    monkeypatch.setattr(
        mod,
        "write_operational_retro",
        lambda root, report: {
            "path": mod.OPERATIONAL_RETRO_RELATIVE_PATH.as_posix(),
            "present": True,
            "sha256": "sha256:retro",
        },
    )
    writes: list[dict[str, object]] = []

    def fake_write_report(
        report: dict[str, object], root: Path = REPO, *, env: object = None
    ) -> Path:
        writes.append(report)
        return tmp_path / mod.RESULT_RELATIVE_PATH.name

    monkeypatch.setattr(mod, "write_report", fake_write_report)
    run_report = mod.run(date="20260812", root=REPO, write=True)
    assert writes and run_report["operational_retro_path_and_hash"]["sha256"] == "sha256:retro"

    no_write_report = mod.run(
        date="20260812",
        root=REPO,
        write=False,
        command_receipts=[{"command": "focused", "exit_code": 0}],
    )
    assert no_write_report["status"] == "complete"
