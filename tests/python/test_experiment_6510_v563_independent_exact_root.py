"""Tests for Exp6510 V563 independent exact root.

Spec refs: REQ-BENCH-6510, SCENARIO-BENCH-6510-DIRECT-IMMUTABLE,
SCENARIO-BENCH-6510-RETIRED-ISOLATION, SCENARIO-BENCH-6510-ATTACKS,
SCENARIO-BENCH-6510-ATOMIC-TERMINAL, SCENARIO-BENCH-6510-VERDICT-CLASS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6510_v563_independent_exact_root as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6510_v563_independent_exact_root.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6510_v563_independent_exact_root.py "
    "-m pytest tests/python/test_experiment_6510_v563_independent_exact_root.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6510_v563_independent_exact_root.py "
    "--fail-under=100 --show-missing"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6510_v563_independent_exact_root --date 20260822"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
FULL_PYTEST_RECEIPT = {
    "command": FULL_PYTEST_COMMAND,
    "exit_code": 3,
    "summary": (
        "repository-wide run stopped after 68 failed, 9638 passed, "
        "8 skipped, 112 warnings, and an xdist worker MemoryError"
    ),
}
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6510_v563_independent_exact_root.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6510_v563_independent_exact_root.json"
)
EXCLUSION_LINT_COMMAND = ".venv/bin/python scripts/exclusion_manifest_lint.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6510_v563_independent_exact_root.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6510_v563_independent_exact_root --validate"
)
GIT_STATUS_COMMAND = "git status --short"

TESTS_RUN = [
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    FULL_PYTEST_RECEIPT,
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": EXCLUSION_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": GIT_STATUS_COMMAND, "exit_code": 0},
]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-BENCH-6510: build a temp artifact without touching tracked results."""

    result_path = tmp_path_factory.mktemp("exp6510") / mod.RESULT_RELATIVE_PATH.name
    return mod.build_artifact(
        repo_root=REPO,
        result_path=result_path,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260822",
    )


def test_req_bench_6510_spec_declares_independent_root_contract() -> None:
    """REQ-BENCH-6510: OpenSpec owns the independent root contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6510") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-BENCH-6510-DIRECT-IMMUTABLE",
        "SCENARIO-BENCH-6510-RETIRED-ISOLATION",
        "SCENARIO-BENCH-6510-ATTACKS",
        "SCENARIO-BENCH-6510-ATOMIC-TERMINAL",
        "SCENARIO-BENCH-6510-VERDICT-CLASS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "artifact_not_updated_past_bootstrap",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_bench_6510_direct_immutable_recomputation(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6510-DIRECT-IMMUTABLE: rows are recomputed from files."""

    receipts = artifact["historical_input_receipts"]
    recomputation = artifact["independent_row_recomputation"]

    assert receipts["exp6504"]["path"] == mod.EXP6504_RELATIVE_PATH.as_posix()
    assert receipts["exp6506"]["path"] == mod.EXP6506_RELATIVE_PATH.as_posix()
    assert receipts["exp6504"]["sha256"].startswith("sha256:")
    assert receipts["exp6506"]["sha256"].startswith("sha256:")
    assert receipts["exp6504"]["json_pointers"] == [
        "/raw_instance_rows",
        "/exact_label_rows",
        "/exact_replay_rows",
        "/split_commitment",
        "/reproducibility_checksum",
    ]
    assert receipts["exp6506"]["json_pointers"] == [
        "/exp6504_row_recomputation",
        "/exp6504_corrigendum",
        "/lineage_decision_rows",
        "/forbidden_dependency_attack_matrix",
        "/v562_exact_branch_ready_score",
    ]

    assert recomputation["exp6504"]["row_replay_passed"] is True
    assert recomputation["exp6504"]["raw_row_count"] == 480
    assert recomputation["exp6504"]["exact_label_row_count"] == 480
    assert recomputation["exp6504"]["exact_replay_row_count"] == 480
    assert recomputation["exp6504"]["label_semantic_match_count"] == 480
    assert recomputation["exp6504"]["replay_failure_count"] == 0
    assert recomputation["exp6504"]["split_hash_matches"] is True
    assert recomputation["exp6504"]["historical_checksum_matches"] is True

    assert recomputation["exp6506"]["contract_recomputed_from_file"] is True
    assert recomputation["exp6506"]["reported_v562_score"] == 1.0
    assert recomputation["exp6506"]["corrected_verdict_class"] == "circular_positive"
    assert recomputation["exp6506"]["artifact_verdict_class"] == "null"
    assert recomputation["exp6506"]["positive_scientific_claim_allowed"] is False
    assert recomputation["exp6506"]["allowed_fields"] == ["exact_label_rows", "raw_instance_rows"]
    assert recomputation["overall_independent_row_checks_passed"] is True


def test_scenario_bench_6510_retired_isolation_and_new_path(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6510-RETIRED-ISOLATION: only the new V563 path is allowed."""

    receipt = artifact["prior_failure_receipt"]
    decisions = {row["scope_id"]: row for row in artifact["lineage_decision_rows"]}

    assert receipt["prior_terminal_result"] == "artifact_not_updated_past_bootstrap"
    assert receipt["exp6506_artifact_not_updated_past_bootstrap_count"] == 3
    assert receipt["exp6506_task_reactivated"] is False
    assert receipt["exp6507_to_exp6509_cascade_preserved"] is True
    assert receipt["material_change"] == "new_id_small_atomic_terminal_artifact"

    assert decisions["v563_exact_branch_counterfactual_path"]["decision"] == "allow"
    assert decisions["v563_exact_branch_counterfactual_path"]["downstream_task"] == (
        "exp6511-exact-branch-counterfactual-dataset-v2"
    )
    for scope in (
        "retired_exp6506_task_id",
        "retired_exp6507_task_id",
        "retired_exp6508_task_id",
        "retired_exp6509_task_id",
        "exp6505_challenge_pool_indirect_use",
        "aggregate_only_exp6504_reuse",
        "positive_class_exact_oracle_claim",
    ):
        assert decisions[scope]["decision"] == "forbid"
        assert decisions[scope]["fail_closed"] is True

    assert mod.structured_dependency_retired_id_violations(artifact) == []
    assert artifact["v563_independent_root_ready_score"] == 1.0
    assert artifact["honest_verdict"].startswith("complete_")


def test_scenario_bench_6510_attacks_and_verdict_class(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6510-ATTACKS/VERDICT-CLASS: shortcuts fail closed."""

    attacks = artifact["retired_dependency_attack_matrix"]

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_attacks_fail_closed"] is True
    assert all(row["fail_closed"] is True for row in attacks["rows"])
    assert all(row["observed_ready_score_if_only_this_attack"] == 0.0 for row in attacks["rows"])
    # REQ-CONDUCTOR-VERDICT-4: the finished replay declares null, not partial.
    assert artifact["verdict_class"] == "null"
    assert artifact["verdict_class"] != "positive"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE

    assert (
        mod.classify_lineage_dependency(
            {
                "scope_id": "missing_hash",
                "dependency_kind": "historical_file_input",
                "source_label": "immutable_exp6504_file",
                "field": "raw_instance_rows",
                "required_hash_present": False,
            }
        )["decision"]
        == "block"
    )
    assert (
        mod.classify_lineage_dependency(
            {
                "scope_id": "retired_alias",
                "dependency_kind": "structured_dependency",
                "source_label": "exp6506-v561-evidence-corrigendum-v562-lineage-lock",
                "field": "v562_exact_branch_ready_score",
                "required_hash_present": True,
            }
        )["decision"]
        == "forbid"
    )
    assert (
        mod.classify_lineage_dependency(
            {
                "scope_id": "new_path",
                "dependency_kind": "structured_dependency",
                "source_label": "exp6510-v563-independent-exact-root",
                "field": "v563_independent_root_ready_score",
                "required_hash_present": True,
                "downstream_task": "exp6511-exact-branch-counterfactual-dataset-v2",
            }
        )["decision"]
        == "allow"
    )


def test_scenario_bench_6510_atomic_terminal_schema_and_validation(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6510-ATOMIC-TERMINAL: schema and checksum validate."""

    result_path = Path(artifact["preconditions_checked"]["result_path"])
    written = json.loads(result_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert artifact["atomic_terminal_write_receipt"]["bootstrap_stub_created"] is False
    assert artifact["atomic_terminal_write_receipt"]["single_terminal_write_path"] is True
    assert artifact["atomic_terminal_write_receipt"]["target_path"] == str(result_path)
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)

    validation_mutations = [
        ("required field set mismatch", lambda item: item.pop("status")),
        ("field_principles mismatch", lambda item: item.__setitem__("field_principles", {})),
        (
            "field_provenance must cover required fields",
            lambda item: item.__setitem__("field_provenance", {}),
        ),
        (
            "verdict_class cannot be positive for oracle readiness",
            lambda item: item.__setitem__("verdict_class", "positive"),
        ),
        # REQ-CONDUCTOR-VERDICT-4 / SCENARIO-CONDUCTOR-VERDICT-5: a ready root
        # may not declare the may-retry class.
        (
            "ready root requires verdict_class null",
            lambda item: item.__setitem__("verdict_class", "partial"),
        ),
        (
            "inference_substrate mismatch",
            lambda item: item.__setitem__("inference_substrate", "live_llm_inference"),
        ),
        (
            "verifier_is_oracle must be true for exact row and hash checks",
            lambda item: item.__setitem__("verifier_is_oracle", False),
        ),
        (
            "retired dependency violation",
            lambda item: next(
                row
                for row in item["lineage_decision_rows"]
                if row["scope_id"] == "v563_exact_branch_counterfactual_path"
            ).__setitem__("source_label", "exp6506-v561-evidence-corrigendum-v562-lineage-lock"),
        ),
        (
            "v563_independent_root_ready_score mismatch",
            lambda item: item.__setitem__("v563_independent_root_ready_score", 0.0),
        ),
        (
            "reproducibility_checksum mismatch",
            lambda item: item.__setitem__("reproducibility_checksum", "sha256:bad"),
        ),
        (
            "honest_verdict lacks terminal prefix",
            lambda item: item.__setitem__("honest_verdict", "ready"),
        ),
    ]
    for expected, mutate in validation_mutations:
        broken = deepcopy(artifact)
        mutate(broken)
        assert expected in mod.validate_artifact(broken)


def test_scenario_bench_6510_fail_closed_defensive_paths(
    artifact: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6510-ATTACKS: defensive paths block unsafe roots."""

    assert mod.sha256_file(tmp_path / "missing.json") == "missing"
    assert (
        mod.classify_lineage_dependency(
            {
                "scope_id": "unknown",
                "dependency_kind": "historical_file_input",
                "source_label": "immutable_unknown_file",
                "field": "row_payload",
                "required_hash_present": True,
            }
        )["reason"]
        == "unknown_dependency_fail_closed"
    )

    violations = mod.structured_dependency_retired_id_violations(
        {
            "lineage_decision_rows": [
                "malformed-row",
                {
                    "decision": "allow",
                    "dependency_kind": "structured_dependency",
                    "source_label": "exp6506-v561-evidence-corrigendum-v562-lineage-lock",
                    "field": "v562_exact_branch_ready_score",
                    "downstream_task": "exp6511-exact-branch-counterfactual-dataset-v2",
                },
            ]
        }
    )
    assert len(violations) == 1

    status, verdict = mod._status_verdict(0.0, {"blocked_reason": "forced_test_block"})
    assert status == "blocked_v563_independent_exact_root"
    assert verdict == "blocked_v563_independent_exact_root: forced_test_block"

    validation_mutations = [
        (
            "prior failure receipt mismatch",
            lambda item: item["prior_failure_receipt"].__setitem__(
                "prior_terminal_result", "complete"
            ),
        ),
        (
            "independent row recomputation failed",
            lambda item: item["independent_row_recomputation"].__setitem__(
                "overall_independent_row_checks_passed", False
            ),
        ),
        (
            "retired dependency attack false accepts",
            lambda item: item["retired_dependency_attack_matrix"].__setitem__(
                "all_attacks_fail_closed", False
            ),
        ),
        (
            "atomic terminal write receipt mismatch",
            lambda item: item["atomic_terminal_write_receipt"].__setitem__(
                "terminal_payload_sha256", ""
            ),
        ),
    ]
    for expected, mutate in validation_mutations:
        broken = deepcopy(artifact)
        mutate(broken)
        assert expected in mod.validate_artifact(broken)

    monkeypatch.setattr(
        mod,
        "recompute_exp6504_direct",
        lambda repo_root, payload: ({"row_replay_passed": False}, []),
    )
    with pytest.raises(ValueError, match="independent row recomputation failed"):
        mod.build_artifact(
            repo_root=REPO,
            result_path=tmp_path / "blocked.json",
            write=False,
            duration_s=0.0,
            tests_run=TESTS_RUN,
            run_date="20260822",
        )


def test_scenario_bench_6510_validation_error_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6510-ATOMIC-TERMINAL: invalid payloads raise."""

    monkeypatch.setattr(mod, "validate_artifact", lambda value: ["forced validation error"])
    monkeypatch.setattr(
        mod,
        "recompute_exp6504_direct",
        lambda repo_root, payload: ({"row_replay_passed": True}, []),
    )
    monkeypatch.setattr(
        mod,
        "recompute_exp6506_contract",
        lambda payload: {"contract_recomputed_from_file": True},
    )

    with pytest.raises(ValueError, match="forced validation error"):
        mod.build_artifact(
            repo_root=REPO,
            result_path=tmp_path / "invalid-build.json",
            write=False,
            duration_s=0.0,
            tests_run=TESTS_RUN,
            run_date="20260822",
        )

    monkeypatch.setattr(
        mod,
        "build_artifact",
        lambda **kwargs: {
            "atomic_terminal_write_receipt": {
                "write_requested": False,
                "terminal_payload_sha256": "",
            }
        },
    )
    with pytest.raises(ValueError, match="forced validation error"):
        mod.run(date="20260822", result_path=tmp_path / "invalid-run.json")


def test_scenario_bench_6510_main_and_validate_roundtrip(tmp_path: Path) -> None:
    """REQ-BENCH-6510: CLI writes and validates the independent root."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name

    assert mod.main(["--date", "20260822", "--result-path", str(result_path)]) == 0
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert payload["v563_independent_root_ready_score"] == 1.0
    assert payload["atomic_terminal_write_receipt"]["target_path"] == str(result_path)
    assert payload["atomic_terminal_write_receipt"]["terminal_payload_sha256"].startswith("sha256:")
    full_receipt = next(
        row for row in payload["tests_run"] if row["command"] == FULL_PYTEST_COMMAND
    )
    assert full_receipt["exit_code"] == 3
    assert "68 failed" in full_receipt["summary"]
