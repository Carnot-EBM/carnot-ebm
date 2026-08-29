"""Tests for Exp6513 V564 terminal handoff.

Spec refs: REQ-BENCH-6513, SCENARIO-BENCH-6513-DIRECT-IMMUTABLE,
SCENARIO-BENCH-6513-ROW-REPLAY, SCENARIO-BENCH-6513-TERMINAL-HISTORY,
SCENARIO-BENCH-6513-RETIRED-ISOLATION, SCENARIO-BENCH-6513-ATTACKS,
SCENARIO-BENCH-6513-ATOMIC-FINAL.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6513_v564_terminal_handoff_contract as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6513_v564_terminal_handoff_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6513_v564_terminal_handoff_contract.py "
    "-m pytest tests/python/test_experiment_6513_v564_terminal_handoff_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6513_v564_terminal_handoff_contract.py "
    "--fail-under=100 --show-missing"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6513_v564_terminal_handoff_contract --date 20260822"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6513_v564_terminal_handoff_contract.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6513_v564_terminal_handoff_contract.json"
)
EXCLUSION_LINT_COMMAND = ".venv/bin/python scripts/exclusion_manifest_lint.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6513_v564_terminal_handoff_contract.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6513_v564_terminal_handoff_contract --validate"
)
GIT_STATUS_COMMAND = "git status --short"

TESTS_RUN = [
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
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
    """REQ-BENCH-6513: build a temp artifact without touching tracked results."""

    result_path = tmp_path_factory.mktemp("exp6513") / mod.RESULT_RELATIVE_PATH.name
    return mod.build_artifact(
        repo_root=REPO,
        result_path=result_path,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260822",
    )


def test_req_bench_6513_spec_declares_handoff_contract() -> None:
    """REQ-BENCH-6513: OpenSpec owns the V564 handoff contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6513") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-BENCH-6513-DIRECT-IMMUTABLE",
        "SCENARIO-BENCH-6513-ROW-REPLAY",
        "SCENARIO-BENCH-6513-TERMINAL-HISTORY",
        "SCENARIO-BENCH-6513-RETIRED-ISOLATION",
        "SCENARIO-BENCH-6513-ATTACKS",
        "SCENARIO-BENCH-6513-ATOMIC-FINAL",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "artifact_not_updated_past_bootstrap",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_bench_6513_direct_inputs_and_row_replay(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6513-DIRECT-IMMUTABLE/ROW-REPLAY: rows own readiness."""

    receipts = {row["input_id"]: row for row in artifact["immutable_input_receipts"]}
    allowed = {row["input_id"]: row for row in artifact["allowed_direct_input_rows"]}
    aggregate = artifact["aggregate_row_recomputation"]

    for input_id in ("exp6504", "exp6506", "exp6510", "exp6512", "conductor_log"):
        assert receipts[input_id]["sha256"].startswith("sha256:")
        assert receipts[input_id]["read_mode"] == "direct_path_and_hash"
    assert receipts["exp6511_missing"]["exists"] is False
    assert receipts["exp6511_missing"]["sha256"] == "missing"

    assert allowed["exp6504_rows"]["counts_as_structured_dependency"] is False
    assert allowed["exp6510_content"]["counts_as_structured_dependency"] is False
    assert allowed["exp6510_content"]["source_task_is_retired"] is True
    assert allowed["exp6510_content"]["eligible_task_dependency"] is False
    assert all(row["read_mode"] == "direct_path_and_hash" for row in allowed.values())

    assert aggregate["exp6504_raw_row_count"] == 480
    assert aggregate["exp6504_exact_label_row_count"] == 480
    assert aggregate["exp6504_exact_replay_row_count"] == 480
    assert aggregate["exp6504_row_replay_passed"] is True
    assert aggregate["exp6510_total_per_unit_row_count"] == 497
    assert aggregate["exp6510_row_type_counts"] == {
        "v563_exp6504_direct_replay": 480,
        "v563_lineage_decision": 10,
        "v563_retired_dependency_attack": 7,
    }
    assert aggregate["exp6510_ready_score_from_rows"] == 1.0
    assert aggregate["exp6510_usable_content"] is True
    assert aggregate["exp6510_eligible_task_dependency"] is False


def test_scenario_bench_6513_terminal_history_preserved(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6513-TERMINAL-HISTORY: old terminal outcomes stay fixed."""

    receipts = {row["receipt_id"]: row for row in artifact["prior_failure_receipts"]}
    tasks = {row["task_id"]: row for row in artifact["historical_task_rows"]}
    event_rows = [
        row for row in artifact["per_unit_rows"] if row["row_type"] == "conductor_terminal_event"
    ]

    assert receipts["exp6506_bootstrap_failures"]["event_count"] == 3
    assert receipts["exp6510_bootstrap_failures"]["event_count"] == 3
    assert receipts["exp6511_missing_dataset"]["exists"] is False
    assert receipts["exp6512_score_zero_block"]["observed_score"] == 0.0
    assert receipts["exp6512_score_zero_block"]["observed_status"].startswith("blocked_")

    assert tasks["exp6504-exact-structural-benchmark-commitment"]["usable_content"] is True
    assert (
        tasks["exp6504-exact-structural-benchmark-commitment"]["eligible_task_dependency"] is False
    )
    assert tasks["exp6510-v563-independent-exact-root"]["source_task_is_retired"] is True
    assert tasks["exp6510-v563-independent-exact-root"]["usable_content"] is True
    assert tasks["exp6510-v563-independent-exact-root"]["eligible_task_dependency"] is False
    assert tasks["exp6511-exact-branch-counterfactual-dataset-v2"]["exists"] is False
    assert tasks["exp6512-branch-dataset-independent-audit"]["terminal_score"] == 0.0
    assert tasks["exp6512-branch-dataset-independent-audit"]["verdict_class"] == "blocked"

    assert len(event_rows) >= 9
    assert all(row["conductor_row_hash"].startswith("sha256:") for row in event_rows)


def test_scenario_bench_6513_retired_isolation_and_attacks(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6513-RETIRED-ISOLATION/ATTACKS: shortcuts fail closed."""

    forbidden = {row["dependency_id"]: row for row in artifact["forbidden_dependency_rows"]}
    attacks = artifact["retired_dependency_attack_matrix"]

    for task_id in mod.RETIRED_OR_INELIGIBLE_TASK_IDS:
        assert forbidden[task_id]["decision"] == "forbid"
        assert forbidden[task_id]["fail_closed"] is True
    assert (
        artifact["preconditions_checked"]["planned_structured_dependency_retired_id_violations"]
        == []
    )
    assert mod.planned_structured_dependency_retired_id_violations(artifact) == []

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_attacks_fail_closed"] is True
    assert attacks["false_accept_count"] == 0
    assert all(row["observed_ready_score_if_only_this_attack"] == 0.0 for row in attacks["rows"])

    assert (
        mod.classify_dependency(
            {
                "dependency_id": "stale_exp6510",
                "dependency_kind": "direct_file_input",
                "source_label": "results/experiment_6510_v563_independent_exact_root.json",
                "required_hash_present": False,
            }
        )["decision"]
        == "block"
    )
    assert (
        mod.classify_dependency(
            {
                "dependency_id": "direct_exp6510",
                "dependency_kind": "direct_file_input",
                "source_label": "results/experiment_6510_v563_independent_exact_root.json",
                "required_hash_present": True,
            }
        )["decision"]
        == "allow"
    )
    assert (
        mod.classify_dependency(
            {
                "dependency_id": "retired_alias",
                "dependency_kind": "structured_dependency",
                "source_label": "v563 independent exact root alias exp6510",
                "required_hash_present": True,
            }
        )["decision"]
        == "forbid"
    )
    assert (
        mod.classify_dependency(
            {
                "dependency_id": "terminal_success_claim",
                "dependency_kind": "interpretation",
                "source_label": "terminal complete means scientific success",
                "required_hash_present": True,
            }
        )["reason"]
        == "terminal_success_is_not_scientific_success"
    )


def test_scenario_bench_6513_atomic_schema_and_validation(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6513-ATOMIC-FINAL: schema and checksum validate."""

    result_path = Path(artifact["preconditions_checked"]["result_path"])
    written = json.loads(result_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    # REQ-CONDUCTOR-VERDICT-4: the finished handoff declares null, not partial.
    assert artifact["verdict_class"] == "null"
    assert artifact["verdict_class"] != "positive"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["v564_handoff_ready_score"] == 1.0
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)

    validation_mutations = [
        ("required field set mismatch", lambda item: item.pop("status")),
        ("field_principles mismatch", lambda item: item.__setitem__("field_principles", {})),
        (
            "field_provenance must cover required fields",
            lambda item: item.__setitem__("field_provenance", {}),
        ),
        (
            "verdict_class cannot be positive",
            lambda item: item.__setitem__("verdict_class", "positive"),
        ),
        # REQ-CONDUCTOR-VERDICT-4 / SCENARIO-CONDUCTOR-VERDICT-5: a ready
        # handoff may not declare the may-retry class.
        (
            "ready handoff requires null verdict_class",
            lambda item: item.__setitem__("verdict_class", "partial"),
        ),
        (
            "inference_substrate mismatch",
            lambda item: item.__setitem__("inference_substrate", "live_llm_inference"),
        ),
        (
            "verifier_is_oracle must be true",
            lambda item: item.__setitem__("verifier_is_oracle", False),
        ),
        (
            "v564_handoff_ready_score mismatch",
            lambda item: item.__setitem__("v564_handoff_ready_score", 0.0),
        ),
        (
            "retired dependency attack false accepts",
            lambda item: item["retired_dependency_attack_matrix"].__setitem__(
                "all_attacks_fail_closed", False
            ),
        ),
        (
            "historical determination not preserved",
            lambda item: item["prior_failure_receipts"][0].__setitem__("event_count", 2),
        ),
        (
            "planned structured dependency retired id violation",
            lambda item: item["preconditions_checked"].__setitem__(
                "planned_structured_dependency_retired_id_violations",
                [{"task_id": "exp6516", "upstream": "exp6510-v563-independent-exact-root"}],
            ),
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


def test_scenario_bench_6513_fail_closed_and_cli_paths(
    artifact: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6513-ATTACKS/ATOMIC-FINAL: defensive branches are closed."""

    assert mod.sha256_file(tmp_path / "missing.json") == "missing"
    assert (
        mod.classify_dependency(
            {
                "dependency_id": "unknown",
                "dependency_kind": "structured_dependency",
                "source_label": "not a known source",
                "required_hash_present": True,
            }
        )["reason"]
        == "unknown_dependency_fail_closed"
    )
    assert (
        mod.classify_dependency(
            {
                "dependency_id": "legacy_alias",
                "dependency_kind": "interpretation",
                "source_label": "retired alias for the prior root",
                "required_hash_present": True,
            }
        )["reason"]
        == "renamed_retired_dependency"
    )

    blocked_status, blocked_verdict = mod.status_and_verdict(
        0.0,
        [{"check": "forced", "observed": "bad"}],
    )
    assert blocked_status == "blocked_v564_terminal_handoff_contract"
    assert blocked_verdict == "blocked_v564_terminal_handoff_contract: forced=bad"
    _, fallback_verdict = mod.status_and_verdict(
        0.0,
        [{"check": "fallback", "observed": "still_closed", "passed": True}],
    )
    assert fallback_verdict.endswith("fallback=still_closed")

    assert mod._planned_structured_dependencies(tmp_path) == []
    roadmap = tmp_path / mod.ROADMAP_RELATIVE_PATH
    roadmap.write_text(
        "tasks:\n"
        "  - bad-list-entry\n"
        "  - id: malformed-gates\n"
        "    gated_on: exp6510-v563-independent-exact-root\n",
        encoding="utf-8",
    )
    assert mod._planned_structured_dependencies(tmp_path) == []

    invalid_score = deepcopy(artifact)
    invalid_score["v564_handoff_ready_score"] = 0.5
    assert "v564_handoff_ready_score must be 0.0 or 1.0" in mod.validate_artifact(invalid_score)

    changed_protected = deepcopy(artifact)
    changed_protected["protected_files_unchanged"]["all_protected_files_unchanged"] = False
    assert "protected files changed" in mod.validate_artifact(changed_protected)

    gate_fail = deepcopy(artifact)
    gate_fail["gate_check_summary"][0]["passed"] = False
    assert "v564_handoff_ready_score mismatch" in mod.validate_artifact(gate_fail)

    invalid_payload = deepcopy(artifact)
    invalid_payload["status"] = "not_terminal"
    invalid_payload["reproducibility_checksum"] = mod.reproducibility_checksum(invalid_payload)
    invalid_path = tmp_path / "invalid-validate.json"
    invalid_path.write_text(json.dumps(invalid_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="status lacks terminal prefix"):
        mod.main(["--validate", "--result-path", str(invalid_path)])

    monkeypatch.setattr(mod, "validate_artifact", lambda value: ["forced validation error"])
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
        lambda **kwargs: {"v564_handoff_ready_score": 1.0},
    )
    with pytest.raises(ValueError, match="forced validation error"):
        mod.run(date="20260822", result_path=tmp_path / "invalid-run.json")


def test_scenario_bench_6513_main_and_validate_roundtrip(tmp_path: Path) -> None:
    """REQ-BENCH-6513: CLI writes and validates the handoff artifact."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name

    assert mod.main(["--date", "20260822", "--result-path", str(result_path)]) == 0
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert payload["v564_handoff_ready_score"] == 1.0
    assert payload["status"] == "complete_v564_terminal_handoff_contract_ready"
    assert payload["preconditions_checked"]["result_path"] == str(result_path)
    assert payload["aggregate_row_recomputation"]["row_type_counts"]["historical_task"] >= 5
    full_receipt = next(
        row for row in payload["tests_run"] if row["command"] == FULL_PYTEST_COMMAND
    )
    assert full_receipt["exit_code"] == 0
