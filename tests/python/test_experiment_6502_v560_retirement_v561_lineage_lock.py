"""Tests for Exp6502 V560 retirement and V561 lineage lock.

Spec refs: REQ-INFRA-6502, SCENARIO-INFRA-6502-RETIREMENT,
SCENARIO-INFRA-6502-CHANGED-SCOPE, SCENARIO-INFRA-6502-DEPENDENCIES,
SCENARIO-INFRA-6502-ATTACKS, SCENARIO-INFRA-6502-PROTECTED,
SCENARIO-INFRA-6502-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6502_v560_retirement_v561_lineage_lock as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6502_v560_retirement_v561_lineage_lock.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6502_v560_retirement_v561_lineage_lock.py "
    "-m pytest tests/python/test_experiment_6502_v560_retirement_v561_lineage_lock.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6502_v560_retirement_v561_lineage_lock.py "
    "--fail-under=100 --show-missing"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6502_v560_retirement_v561_lineage_lock.py"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6502_v560_retirement_v561_lineage_lock "
    "--date 20260822"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6502_v560_retirement_v561_lineage_lock.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6502_v560_retirement_v561_lineage_lock.json"
)
DOC_COMMAND = (
    ".venv/bin/python -c \"from pathlib import Path; assert Path('ops/e2e-test-plan.md').exists()\""
)
TESTS_RUN = [
    {"command": TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": DOC_COMMAND, "exit_code": 0},
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


def test_req_infra_6502_spec_declares_lineage_lock_contract() -> None:
    """REQ-INFRA-6502: OpenSpec owns the Exp6502 ledger contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6502") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-INFRA-6502-RETIREMENT",
        "SCENARIO-INFRA-6502-CHANGED-SCOPE",
        "SCENARIO-INFRA-6502-DEPENDENCIES",
        "SCENARIO-INFRA-6502-ATTACKS",
        "SCENARIO-INFRA-6502-PROTECTED",
        "SCENARIO-INFRA-6502-ARTIFACT",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_infra_6502_retires_v560_claim_surfaces(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6502-RETIREMENT: rows decide every V560 surface."""

    artifact = _artifact(tmp_path)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())
    decisions = {row["claim_surface"]: row for row in artifact["decision_rows"]}

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_v560_retirement_v561_lineage_locked"
    assert artifact["honest_verdict"].startswith("complete_v560_retirement_v561_lineage_lock")
    assert artifact["verdict_class"] == "null"
    assert artifact["v561_lineage_lock_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True

    receipts = {row["experiment_id"]: row for row in artifact["v560_artifact_receipts"]}
    assert len(receipts) == 14
    assert receipts["exp6494"]["exists"] is False
    assert receipts["exp6494"]["sha256"] == "missing"
    assert receipts["exp6501"]["sha256"].startswith("sha256:")
    assert all(
        row["sha256"].startswith("sha256:") for row in receipts.values() if row["exists"] is True
    )

    assert decisions["exact_solver_trajectory_contract"]["disposition"] == "reuse"
    assert decisions["exact_solver_trajectory_contract"]["retirement_status"] == "reusable"
    assert (
        "exact_solver_trajectory_commitment"
        in decisions["exact_solver_trajectory_contract"]["allowed_reuse"]
    )

    learned = decisions["learned_trajectory_energy"]
    assert learned["verdict"] == "disqualified"
    assert learned["observed_field"] == "trajectory_signal_ready_score_from_rows"
    assert learned["observed_value"] == 0.0
    assert learned["disposition"] == "retire"
    assert learned["retirement_status"] == "retired"
    assert learned["allowed_reuse"] == []

    factor = decisions["factor_causal_value"]
    assert factor["verdict"] == "null"
    assert factor["observed_value"] == 0.0
    assert factor["retirement_status"] == "retired"

    decomposed = decisions["decomposed_energy_checker_routing"]
    assert decomposed["verdict"] == "blocked"
    assert decomposed["disposition"] == "retire"
    assert decomposed["retirement_status"] == "retired"

    csl = decisions["continuous_factor_learning"]
    assert csl["observed_field"] == "continuous_self_learning_ready_score_from_rows"
    assert csl["observed_value"] == 0.0
    assert csl["retirement_status"] == "retired"

    lifecycle = decisions["factor_pool_lifecycle_controls"]
    assert lifecycle["disposition"] == "reuse"
    assert lifecycle["retirement_status"] == "mechanism_reusable"
    assert "bounded_lifecycle_receipts" in lifecycle["allowed_reuse"]

    arc = decisions["arc_energy_policy"]
    assert arc["observed_field"] == "arc_energy_alignment_ready_score_from_rows"
    assert arc["observed_value"] == 0.0
    assert arc["retirement_status"] == "deferred_until_fresh_alignment"

    hardware = decisions["hardware_acceleration_claim"]
    assert hardware["observed_field"] == "hardware_claim_eligible"
    assert hardware["observed_value"] is False
    assert hardware["retirement_status"] == "deferred_until_authenticated_hardware"


def test_scenario_infra_6502_changed_scope_and_dependency_lock(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6502-CHANGED-SCOPE/DEPENDENCIES: V561 is fresh."""

    artifact = _artifact(tmp_path)
    retired = {row["scope_id"]: row for row in artifact["retired_scope_definition"]["scopes"]}
    lineage = artifact["allowed_v561_lineage"]
    aggregate = artifact["aggregate_row_recomputation"]

    assert set(retired) == {
        "learned_trajectory_energy",
        "factor_causal_value",
        "decomposed_energy_checker_routing",
        "factor_pool_learning",
        "arc_energy_policy",
        "hardware_acceleration_claims",
    }
    assert retired["learned_trajectory_energy"]["v561_reuse_allowed"] is False
    assert retired["factor_pool_learning"]["allowed_reuse"] == [
        "transactional_update_receipts",
        "rollback_and_support_checks",
    ]
    assert retired["factor_pool_learning"]["forbidden_reuse"] == [
        "factor_creation",
        "factor_pool_policy",
        "held_future_benefit_claim",
    ]

    assert lineage["lineage_id"] == "v561_exact_sat_csp_structural_branch_advice"
    assert lineage["new_task_distribution"] == "exact_sat_csp"
    assert lineage["allowed_methods"] == [
        "new_exact_sat_csp_distribution",
        "solver_native_structural_advice",
        "exact_branch_counterfactual_labels",
        "fixed_feature_weight_updates",
        "fixed_width_mapping",
    ]
    assert lineage["acceptance_authority"] == [
        "exact_cdcl_solver",
        "exact_csp_repair",
        "executable_validity_check",
    ]
    assert lineage["learned_advice_can_accept_solution"] is False
    assert lineage["retired_upstream_experiment_ids"] == mod.RETIRED_UPSTREAM_EXPERIMENT_IDS

    assert aggregate["decision_row_count"] == len(artifact["decision_rows"])
    assert aggregate["every_decision_row_recomputed"] is True
    assert aggregate["all_forbidden_reuse_attacks_fail_closed"] is True
    assert aggregate["no_v561_task_depends_on_retired_upstream_experiment_id"] is True
    assert aggregate["capstone_claims_recomputed_from_rows"] is True
    assert aggregate["claim_eligibility"] == {
        "trajectory_energy_claim_eligible": False,
        "continuous_learning_claim_eligible": False,
        "arc_policy_claim_eligible": False,
        "hardware_claim_eligible": False,
    }
    assert (
        artifact["gate_check_summary"]["checks"][
            "no_v561_task_depends_on_retired_upstream_experiment_id"
        ]
        is True
    )


def test_scenario_infra_6502_attacks_and_protected_files(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6502-ATTACKS/PROTECTED: attacks and hashes close gates."""

    artifact = _artifact(tmp_path)
    attacks = {row["attack_id"]: row for row in artifact["forbidden_reuse_attack_matrix"]}
    protected = artifact["protected_files_unchanged"]

    assert set(attacks) == {
        "renamed_learned_trajectory_energy",
        "hidden_retired_task_dependency",
        "post_hoc_corpus_repair",
        "generated_answer_transport",
        "nl_to_constraintir_reprompting",
        "arc_policy_edit",
        "hardware_claim_laundering",
        "mechanism_claim_laundering",
    }
    assert all(row["fail_closed"] is True for row in attacks.values())
    assert all(row["allowed_into_v561"] is False for row in attacks.values())
    assert attacks["hidden_retired_task_dependency"]["observed_value"] == 0
    assert attacks["hardware_claim_laundering"]["observed_blocker"] == (
        "no_authenticated_local_special_hardware_evidence"
    )

    assert protected["active_roadmap_and_conductor_unchanged"] is True
    assert protected["changed_paths"] == []
    assert protected["files"]["research-roadmap.yaml"]["unchanged"] is True
    assert protected["files"]["scripts/research_conductor.py"]["unchanged"] is True

    pre = artifact["preconditions_checked"]
    assert pre["planning_date"] == "20260822"
    assert pre["compute"]["no_gpu_required"] is True
    assert pre["network"]["network_required"] is False
    assert pre["exclusion_manifest"]["loaded"] is True
    assert pre["preconditions_ready"] is True


def test_scenario_infra_6502_artifact_validation_and_edges(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6502-ARTIFACT: malformed ledgers fail closed."""

    artifact = _artifact(tmp_path / "clean")
    assert mod.tests_run_receipts(TESTS_RUN) == TESTS_RUN
    assert all(row["exit_code"] == 0 for row in mod.tests_run_receipts(None))
    assert mod.sha256_file(tmp_path / "missing.json") == "missing"
    assert mod._json_pointer({"a": {"b": 2}}, "/a/b") == 2
    assert mod._json_pointer({"a": 1}, "/a/b") is None
    assert mod._json_pointer({"a": 1}, "a") is None
    assert mod._rows_from({"per_unit_rows": [{"a": 1}, "bad"]}) == [{"a": 1}]
    assert mod._rows_from({"rows": {"rows": [{"b": 2}, "bad"]}}, "rows") == [{"b": 2}]
    assert mod._rows_from({}, "missing") == []
    assert mod._capstone_manifest_paths(tmp_path / "no-capstone") == mod.FALLBACK_V560_PATHS

    fake_repo = tmp_path / "fake-repo"
    (fake_repo / "results").mkdir(parents=True)
    (fake_repo / mod.CAPSTONE_RELATIVE_PATH).write_text(
        json.dumps({"milestone_manifest_rows": ["bad", {"experiment_id": "exp6494"}]}),
        encoding="utf-8",
    )
    assert mod._capstone_manifest_paths(fake_repo)["exp6494"] == mod.FALLBACK_V560_PATHS["exp6494"]

    assert mod.scan_v561_dependency_rows(
        {
            "tasks": [
                "bad-task",
                {"id": "exp6503-ok", "gated_on": [{"upstream": "exp6502-lock"}]},
                {"id": "exp6504-bad", "gated_on": [{"upstream": "exp6490-old"}]},
                {"id": "exp6505-ignored", "gated_on": ["bad"]},
            ]
        },
        retired_ids=("exp6490",),
    ) == [
        {
            "row_type": "dependency_scan",
            "task_id": "exp6504-bad",
            "upstream": "exp6490-old",
            "upstream_experiment_id": "exp6490",
            "retired_dependency": True,
        }
    ]

    bad = deepcopy(artifact)
    del bad["status"]
    _with_checksum(bad)
    assert any("missing required fields" in error for error in mod.validate_artifact(bad))

    bad = deepcopy(artifact)
    bad["unexpected"] = True
    bad["field_principles"] = {}
    bad["field_provenance"] = {}
    bad["verdict_class"] = "bad"
    bad["inference_substrate"] = "wrong"
    bad["verifier_is_oracle"] = False
    _with_checksum(bad)
    errors = mod.validate_artifact(bad)
    assert any("unexpected fields" in error for error in errors)
    assert "field_principles must cover exactly required fields" in errors
    assert "field_provenance must cover exactly required fields" in errors
    assert "verdict_class outside closed enum" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be true" in errors

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "complete: wrong prefix"
    _with_checksum(bad)
    assert "honest_verdict lacks accepted Exp6502 prefix" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["v561_lineage_lock_ready_score"] = 0.0
    bad["gate_check_summary"]["all_gates_passed"] = True
    _with_checksum(bad)
    assert "ready score and gate summary disagree" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["v560_artifact_receipts"] = []
    _with_checksum(bad)
    assert "v560_artifact_receipts must contain 14 rows" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["decision_rows"][0]["recomputed"] = False
    _with_checksum(bad)
    assert "decision_rows must all recompute" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["forbidden_reuse_attack_matrix"][0]["fail_closed"] = False
    _with_checksum(bad)
    assert "forbidden reuse attacks must fail closed" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["aggregate_row_recomputation"]["no_v561_task_depends_on_retired_upstream_experiment_id"] = (
        False
    )
    _with_checksum(bad)
    assert "retired V560 dependency detected" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:" + "1" * 64
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)

    invalid = tmp_path / "invalid.json"
    invalid.write_text("[1]\n", encoding="utf-8")
    assert "artifact must be a JSON object" in mod.validate_artifact(invalid)[0]

    with pytest.raises(ValueError, match="forced validation error"):
        original = mod.validate_artifact
        try:
            mod.validate_artifact = lambda _value: ["forced validation error"]  # type: ignore[method-assign]
            mod.build_artifact(repo_root=REPO, write=False, duration_s=1.0)
        finally:
            mod.validate_artifact = original  # type: ignore[method-assign]

    rc = mod.main(["--date", "20260822", "--output", str(tmp_path / "main.json")])
    assert rc == 0
    written = json.loads((tmp_path / "main.json").read_text())
    assert written["preconditions_checked"]["planning_date"] == "20260822"
