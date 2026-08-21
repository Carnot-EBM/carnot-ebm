"""Tests for Exp6488 V559 decision ledger.

Spec refs: REQ-INFRA-6488, SCENARIO-INFRA-6488-RECOMPUTE,
SCENARIO-INFRA-6488-DISPOSITIONS, SCENARIO-INFRA-6488-LINEAGE,
SCENARIO-INFRA-6488-ATTACKS, SCENARIO-INFRA-6488-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6488_v559_decision_ledger as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6488_v559_decision_ledger.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6488_v559_decision_ledger.py "
    "-m pytest tests/python/test_experiment_6488_v559_decision_ledger.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6488_v559_decision_ledger.py "
    "--fail-under=100 --show-missing"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6488_v559_decision_ledger.py"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6488_v559_decision_ledger --date 20260821"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6488_v559_decision_ledger.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6488_v559_decision_ledger.json"
)
E2E_COMMAND = ".venv/bin/python -c \"from pathlib import Path; assert Path('ops/e2e-test-plan.md').exists()\""
TESTS_RUN = [
    {"command": TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_COMMAND, "exit_code": 0},
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": E2E_COMMAND, "exit_code": 0},
]


def _artifact(tmp_path: Path) -> dict[str, Any]:
    return mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
    )


def _with_checksum(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    return artifact


def test_req_infra_6488_spec_declares_lineage_ledger_contract() -> None:
    """REQ-INFRA-6488: OpenSpec owns the Exp6488 ledger contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6488") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-INFRA-6488-RECOMPUTE",
        "SCENARIO-INFRA-6488-DISPOSITIONS",
        "SCENARIO-INFRA-6488-LINEAGE",
        "SCENARIO-INFRA-6488-ATTACKS",
        "SCENARIO-INFRA-6488-ARTIFACT",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_infra_6488_recomputes_v559_and_freezes_scope(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6488-RECOMPUTE/DISPOSITIONS: rows drive decisions."""

    artifact = _artifact(tmp_path)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())
    decisions = {row["claim_surface"]: row for row in artifact["decision_rows"]}
    raw = artifact["aggregate_row_recomputation"]["exp6486_raw_rows"]
    shortcuts = artifact["aggregate_row_recomputation"]["exp6487_shortcut_replay"]
    missing = artifact["aggregate_row_recomputation"]["missing_evidence_replay"]

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_v559_decision_ledger"
    assert artifact["honest_verdict"].startswith("complete_v560_lineage_lock:")
    assert artifact["v560_lineage_lock_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert len(artifact["v559_artifact_receipts"]) == 5
    assert all(row["sha256"].startswith("sha256:") for row in artifact["v559_artifact_receipts"])

    assert decisions["source_map_and_evidence_boundary"]["disposition"] == "reuse"
    assert decisions["non_generation_representation_receipt_contract"]["disposition"] == "reuse"
    assert decisions["online_cache_transition_eprocess_contract"]["disposition"] == "reuse"
    assert decisions["forced_candidate_representation_rows"]["disposition"] == "freeze"
    assert decisions["forced_candidate_representation_selector_scope"]["disposition"] == "retire"
    assert decisions["representation_integrity_audit"]["disposition"] == "informational_only"
    assert decisions["forced_candidate_representation_selector_scope"]["selector_eligible"] is False

    assert raw["raw_record_count"] == 432
    assert raw["pair_count"] == 288
    assert raw["label_counts"] == {"correct": 144, "wrong": 288}
    assert raw["split_counts"] == {"calibration": 54, "development": 162, "held": 216}
    assert raw["family_counts"] == {
        "gemma4_26b_a4b_it": 144,
        "gemma4_31b_it": 144,
        "qwen3_6_35b_a3b": 144,
    }
    assert raw["raw_hashes_match_manifest"] is True

    assert shortcuts["surviving_shortcuts"] == [
        "candidate_identifier_length",
        "candidate_identity",
        "row_order_modulo_pair",
    ]
    for name in shortcuts["surviving_shortcuts"]:
        rows = [row for row in shortcuts["shortcut_rows"] if row["control_name"] == name]
        assert len(rows) == 3
        assert all(row["balanced_accuracy"] == 1.0 for row in rows)
    assert missing == {
        "candidate_length_unavailable_from_raw_rows": 432,
        "prompt_length_unavailable_from_raw_rows": 432,
        "token_length_unavailable_from_raw_rows": 432,
    }

    retired = artifact["retired_scope_definition"]
    assert retired["selector_eligible"] is False
    assert retired["v559_forced_candidate_rows_eligible_for_v560_selector"] == 0
    assert "candidate_identifier_length" in retired["disqualifying_shortcuts"]
    assert artifact["protected_files_unchanged"]["active_roadmap_and_conductor_unchanged"] is True


def test_scenario_infra_6488_lineage_and_attacks_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6488-LINEAGE/ATTACKS: V560 scope is exact-only."""

    artifact = _artifact(tmp_path)
    lineage = artifact["allowed_v560_lineage"]
    attacks = {row["attack_name"]: row for row in artifact["forbidden_reuse_attack_matrix"]}

    assert lineage["lineage_id"] == "v560_exact_solver_trajectory"
    assert lineage["prospective_exact_solver_states"] is True
    assert lineage["early_to_final_persistence_labels"] is True
    assert lineage["identity_free_features_required"] is True
    assert lineage["exact_replay_required"] is True
    assert lineage["hidden_state_selector_reuse_allowed"] is False
    assert lineage["allowed_feature_families"] == [
        "solver_state_observables",
        "exact_constraint_residuals",
        "chronological_event_features",
    ]

    assert set(attacks) == {
        "relabel_v559_candidates",
        "repair_lengths_post_hoc",
        "filter_shortcut_rows",
        "reuse_fitted_representation_transform",
        "cite_contract_readiness_as_scientific_signal",
    }
    assert all(row["fail_closed"] is True for row in attacks.values())
    assert all(row["allowed_into_v560"] is False for row in attacks.values())
    assert attacks["filter_shortcut_rows"]["observed_blocker"] == "shortcut_rows_are_evidence"
    assert artifact["aggregate_row_recomputation"]["forbidden_reuse_attack_fail_closed_count"] == 5
    assert artifact["aggregate_row_recomputation"]["all_v559_dispositions_recomputed"] is True
    assert artifact["aggregate_row_recomputation"]["no_v559_forced_candidate_row_selector_eligible"] is True


def test_scenario_infra_6488_artifact_validation_and_blockers(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6488-ARTIFACT: blockers and malformed ledgers fail."""

    clean = _artifact(tmp_path / "clean")
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(bad_json)
    loose_raw = tmp_path / "loose.json"
    loose_raw.write_text("{}\n", encoding="utf-8")
    paths, entries = mod._manifest_paths({"raw_vector_manifest": {"vectors": [{"path": str(loose_raw)}]}})
    assert paths == [loose_raw]
    assert entries[str(loose_raw)]["path"] == str(loose_raw)
    assert mod._candidate_kind({"candidate_kind": "fallback"}, {"path_candidate_kind": "path"}) == "fallback"
    assert mod._label_from_kind("unlabeled") is None
    empty_raw, empty_shortcuts, empty_missing = mod.exp6486_raw_replay({})
    assert empty_raw["raw_record_count"] == 0
    assert empty_shortcuts["surviving_shortcuts"] == []
    assert empty_missing == {}
    assert mod._receipt_tests_pass([{"exit_code": 0}]) is True

    missing_path = tmp_path / "missing-exp6487.json"
    blocked = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked.json",
        artifact_paths={**mod.DEFAULT_ARTIFACT_PATHS, "exp6487": missing_path},
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
    )

    assert blocked["status"] == "blocked_v559_decision_ledger"
    assert blocked["v560_lineage_lock_ready_score"] == 0.0
    assert blocked["honest_verdict"].startswith("blocked_v560_lineage_lock:")
    assert "missing_v559_artifact:exp6487" in blocked["gate_check_summary"]["failed_checks"]
    assert mod.validate_artifact(blocked) == []

    missing_field = deepcopy(clean)
    del missing_field["status"]
    with pytest.raises(ValueError, match="status"):
        mod.assert_valid_artifact(missing_field)

    bad_checksum = deepcopy(clean)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.assert_valid_artifact(bad_checksum)

    bad_principles = deepcopy(clean)
    bad_principles["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        mod.assert_valid_artifact(bad_principles)

    bad_provenance = deepcopy(clean)
    bad_provenance["field_provenance"] = {}
    with pytest.raises(ValueError, match="field_provenance"):
        mod.assert_valid_artifact(bad_provenance)

    bad_oracle = deepcopy(clean)
    bad_oracle["verifier_is_oracle"] = False
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.assert_valid_artifact(bad_oracle)

    bad_status = deepcopy(clean)
    bad_status["status"] = "complete_wrong"
    with pytest.raises(ValueError, match="status"):
        mod.assert_valid_artifact(bad_status)

    bad_verdict = deepcopy(clean)
    bad_verdict["honest_verdict"] = "complete_wrong"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.assert_valid_artifact(bad_verdict)

    bad_attack = deepcopy(clean)
    bad_attack["forbidden_reuse_attack_matrix"][0]["fail_closed"] = False
    _with_checksum(bad_attack)
    with pytest.raises(ValueError, match="v560_lineage_lock_ready_score"):
        mod.assert_valid_artifact(bad_attack)

    bad_decision = deepcopy(clean)
    bad_decision["decision_rows"][0]["disposition"] = "launder"
    _with_checksum(bad_decision)
    with pytest.raises(ValueError, match="decision_rows"):
        mod.assert_valid_artifact(bad_decision)

    bad_substrate = deepcopy(clean)
    bad_substrate["inference_substrate"] = "live_llm"
    bad_substrate["v560_lineage_lock_ready_score"] = mod.v560_lineage_lock_ready_score(
        bad_substrate
    )
    _with_checksum(bad_substrate)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.assert_valid_artifact(bad_substrate)
