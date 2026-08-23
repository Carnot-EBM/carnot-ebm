"""Tests for Exp6522 chronological exact-conflict self-learning.

Spec refs: REQ-STORE-6522, SCENARIO-STORE-6522-SEALING,
SCENARIO-STORE-6522-MATCHED-DOSE, SCENARIO-STORE-6522-LEARNING-ACTIONS,
SCENARIO-STORE-6522-FUTURE-SUPPORT, SCENARIO-STORE-6522-PREFIX-RETENTION,
SCENARIO-STORE-6522-SAFETY, SCENARIO-STORE-6522-RESTART-ROLLBACK-CAPACITY,
SCENARIO-STORE-6522-SEQUENTIAL-EVIDENCE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6522_chronological_conflict_self_learning as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6522_chronological_conflict_self_learning.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6522_chronological_conflict_self_learning.py "
    "-m pytest tests/python/test_experiment_6522_chronological_conflict_self_learning.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6522_chronological_conflict_self_learning.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6522_chronological_conflict_self_learning.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6522_chronological_conflict_self_learning.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6522_chronological_conflict_self_learning.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6522_chronological_conflict_self_learning --date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6522_chronological_conflict_self_learning --validate"
)
GIT_STATUS_COMMAND = "git status --short"

TESTS_RUN = [
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": GIT_STATUS_COMMAND, "exit_code": 0},
]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-STORE-6522: build the artifact without writing tracked results."""

    root = tmp_path_factory.mktemp("exp6522")
    return mod.build_artifact(
        repo_root=REPO,
        result_path=root / mod.RESULT_RELATIVE_PATH.name,
        work_root=root / "work",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )


def test_req_store_6522_spec_declares_chronological_contract() -> None:
    """REQ-STORE-6522: OpenSpec owns the chronological learning contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-STORE-6522") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-STORE-6522-SEALING",
        "SCENARIO-STORE-6522-MATCHED-DOSE",
        "SCENARIO-STORE-6522-LEARNING-ACTIONS",
        "SCENARIO-STORE-6522-FUTURE-SUPPORT",
        "SCENARIO-STORE-6522-PREFIX-RETENTION",
        "SCENARIO-STORE-6522-SAFETY",
        "SCENARIO-STORE-6522-RESTART-ROLLBACK-CAPACITY",
        "SCENARIO-STORE-6522-SEQUENTIAL-EVIDENCE",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "continuous_self_learning_candidate_score",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_store_6522_artifact_schema_and_positive_gate(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-STORE-6522-SEQUENTIAL-EVIDENCE: scores replay from rows."""

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_positive_chronological_conflict_self_learning"
    assert artifact["honest_verdict"].startswith("complete_positive_")
    assert artifact["verdict_class"] == "positive"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["exact_solver_is_release_authority"] is True
    assert artifact["csl_execution_complete_score"] == 1.0
    assert artifact["continuous_self_learning_candidate_score"] == 1.0
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    gate = artifact["upstream_gate_receipt"]
    assert gate["path"] == mod.EXP6521_RELATIVE_PATH.as_posix()
    assert gate["expected_value"] == 1.0
    assert gate["observed_value"] == 1.0
    assert gate["gate_passed"] is True
    assert gate["artifact_sha256"].startswith("sha256:")
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert artifact["preconditions_checked"]["run_date"] == "20260823"
    assert artifact["preconditions_checked"]["solver_versions"]["exact_solver"].endswith("_v1")


def test_scenario_store_6522_sealed_stream_and_action_rows(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-STORE-6522-LEARNING-ACTIONS: online actions are visible."""

    commitment = artifact["chronological_stream_commitment"]
    assert commitment["held_future_boundary_index"] == 8
    assert commitment["thresholds_frozen_before_execution"] is True
    assert commitment["uses_future_outcomes_for_stream"] is False
    assert commitment["stream_hash"].startswith("sha256:")
    assert {
        "refinement_chain",
        "unrelated_query",
        "recurrence_after_gap",
        "distribution_shift",
        "corruption_injection",
        "held_future_suffix",
    } <= set(commitment["coverage_tags"])

    action_names = {row["action"] for row in artifact["lifecycle_action_rows"]}
    assert {
        "propose",
        "validate",
        "commit",
        "use",
        "abstain",
        "evict",
        "rollback",
        "quarantine",
        "fallback",
    } <= action_names
    assert all(row["terminal"] is True for row in artifact["lifecycle_action_rows"])

    planned = set(artifact["arm_and_dose_contract"]["arm_names"])
    expected_store_rows = commitment["event_count"] * len(planned)
    assert len(artifact["store_hash_rows"]) == expected_store_rows
    assert all(row["store_hash_before"] and row["store_hash_after"] for row in artifact["store_hash_rows"])


def test_scenario_store_6522_matched_dose_equality_and_benefit(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-STORE-6522-MATCHED-DOSE/FUTURE-SUPPORT: benefit is charged."""

    contract = artifact["arm_and_dose_contract"]
    opportunity_counts = {row["arm"]: row["opportunity_count"] for row in contract["dose_rows"]}
    assert len(set(opportunity_counts.values())) == 1
    assert set(opportunity_counts) == set(contract["arm_names"])
    assert all(row["lookup_charge"] == contract["lookup_charge"] for row in contract["dose_rows"])
    assert all(row["mapping_charge"] == contract["mapping_charge"] for row in contract["dose_rows"])

    assert all(row["exact_answer_equal"] is True for row in artifact["exact_answer_equality_rows"])
    support = {row["arm"]: row for row in artifact["held_future_support_rows"]}
    assert support["valid_unbounded_reuse"]["charged_benefit_vs_scratch"] > 0
    assert support["valid_bounded_reuse"]["charged_benefit_vs_scratch"] > 0
    assert support["valid_unbounded_reuse"]["charged_benefit_vs_frozen_empty"] > 0
    assert support["valid_bounded_reuse"]["positive_chain_count"] >= 2
    assert support["scratch"]["charged_benefit_vs_scratch"] == 0

    used_rows = [
        row
        for row in artifact["immediate_metric_rows"]
        if row["arm"] == "valid_bounded_reuse" and row["memory_used"]
    ]
    assert used_rows
    assert all(row["current_query_utility"] > 0 for row in used_rows)


def test_scenario_store_6522_retention_interference_capacity_and_safety(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-STORE-6522-SAFETY/PREFIX-RETENTION: attacks fail closed."""

    assert all(row["retention_within_margin"] for row in artifact["prefix_retention_rows"])
    assert min(row["support_after"] for row in artifact["prefix_retention_rows"]) >= 1.0

    assert artifact["interference_rows"]
    assert all(row["unsafe_unrelated_reuse_count"] == 0 for row in artifact["interference_rows"])
    assert all(row["exact_answer_equal"] is True for row in artifact["interference_rows"])

    capacity = artifact["capacity_restart_rollback_rows"]
    assert any(row["check"] == "capacity_eviction" and row["passed"] for row in capacity)
    assert any(row["check"] == "restart_parity" and row["passed"] for row in capacity)
    assert any(row["check"] == "rollback_parity" and row["passed"] for row in capacity)

    attacks = artifact["invalid_reuse_attack_rows"]
    expected = {
        "replay_leakage",
        "future_aware_eviction",
        "unequal_opportunities",
        "hidden_full_set_validation",
        "unsafe_unrelated_reuse",
        "restart_drift",
        "rollback_drift",
        "support_collapse",
        "one_chain_benefit",
        "aggregate_only_claim",
        "relaxed_query",
        "unrelated_query",
        "schema_mismatch",
        "invalid_replay",
    }
    assert expected <= {row["attack_id"] for row in attacks}
    assert all(row["vetoed"] is True and row["passed"] is True for row in attacks)
    assert sum(row["durable_write_performed"] for row in attacks) == 0
    assert sum(row["unsafe_use_performed"] for row in attacks) == 0

    seq = artifact["sequential_evidence"]
    assert seq["held_future_unread_until_boundary"] is True
    assert seq["decisions_use_only_prior_store_hash"] is True
    assert seq["aggregate_only_claim_blocked"] is True


def test_scenario_store_6522_validation_and_cli_roundtrip(
    tmp_path: Path,
    artifact: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-STORE-6522-SEALING: CLI writes and validates the artifact."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    work_root = tmp_path / "work"
    assert (
        mod.main(
            [
                "--date",
                "20260823",
                "--result-path",
                str(result_path),
                "--work-root",
                str(work_root),
            ]
        )
        == 0
    )
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["continuous_self_learning_candidate_score"] == 1.0

    mutations = [
        ("required field set mismatch", lambda item: item.pop("status")),
        ("status lacks terminal prefix", lambda item: item.__setitem__("status", "running")),
        (
            "honest_verdict lacks terminal prefix",
            lambda item: item.__setitem__("honest_verdict", "running"),
        ),
        (
            "verdict_class outside Exp6522 enum",
            lambda item: item.__setitem__("verdict_class", "circular_positive"),
        ),
        (
            "inference_substrate mismatch",
            lambda item: item.__setitem__("inference_substrate", "live_llm"),
        ),
        (
            "verifier_is_oracle must be false",
            lambda item: item.__setitem__("verifier_is_oracle", True),
        ),
        (
            "exact solver release authority missing",
            lambda item: item.__setitem__("exact_solver_is_release_authority", False),
        ),
        ("field_principles mismatch", lambda item: item.__setitem__("field_principles", {})),
        (
            "field_provenance must cover required fields",
            lambda item: item.__setitem__("field_provenance", {}),
        ),
        (
            "csl_execution_complete_score mismatch",
            lambda item: item.__setitem__("csl_execution_complete_score", 0.0),
        ),
        (
            "continuous_self_learning_candidate_score mismatch",
            lambda item: item.__setitem__("continuous_self_learning_candidate_score", 0.0),
        ),
        (
            "unsafe write or use detected",
            lambda item: item["invalid_reuse_attack_rows"][0].__setitem__(
                "durable_write_performed", True
            ),
        ),
        (
            "exact answer drift detected",
            lambda item: item["exact_answer_equality_rows"][0].__setitem__(
                "exact_answer_equal", False
            ),
        ),
        (
            "protected files changed",
            lambda item: item["protected_files_unchanged"].__setitem__(
                "all_protected_files_unchanged", False
            ),
        ),
        (
            "reproducibility_checksum mismatch",
            lambda item: item.__setitem__("reproducibility_checksum", "sha256:bad"),
        ),
    ]
    for expected, mutate in mutations:
        broken = deepcopy(artifact)
        mutate(broken)
        assert expected in mod.validate_artifact(broken)

    blocked = deepcopy(artifact)
    blocked["upstream_gate_receipt"]["gate_passed"] = False
    blocked["reproducibility_checksum"] = mod.reproducibility_checksum(blocked)
    assert "upstream gate failed" in mod.validate_artifact(blocked)

    invalid_path = tmp_path / "invalid.json"
    invalid = deepcopy(payload)
    invalid["status"] = "running"
    invalid["reproducibility_checksum"] = mod.reproducibility_checksum(invalid)
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="status lacks terminal prefix"):
        mod.main(["--validate", "--result-path", str(invalid_path)])

    assert mod._read_json(tmp_path / "missing.json") == {}
    unsat = mod._solve_accounting(mod.ExactQuery(1, ((1,), (-1,))), ())
    assert unsat["exact_status"] == "unsat"
    assert mod._find_reusable_record(None, mod.ExactQuery(1, ((1,),))) is None

    no_restart_event = mod.StreamEvent(
        "no_restart",
        1,
        "prefix",
        "no_restart_chain",
        "query",
        mod.ExactQuery(1, ((1,),)),
        None,
        0.0,
        ("refinement_chain",),
    )
    restart_rows = mod._run_arm("restart", (no_restart_event,), tmp_path / "no-restart")
    rollback_rows = mod._run_arm("rollback", (no_restart_event,), tmp_path / "no-rollback")
    assert any(row["check"] == "restart_parity" and row["passed"] is False for row in restart_rows["capacity_restart_rollback_rows"])
    assert any(row["check"] == "rollback_parity" and row["passed"] is False for row in rollback_rows["capacity_restart_rollback_rows"])

    monkeypatch.setattr(
        mod,
        "upstream_gate_receipt",
        lambda repo_root: {
            "gate_id": "mock",
            "path": mod.EXP6521_RELATIVE_PATH.as_posix(),
            "absolute_path": str(tmp_path / "mock.json"),
            "artifact_sha256": "sha256:" + "1" * 64,
            "exists": True,
            "field": "conflict_memory_controller_ready_score",
            "expected_value": 1.0,
            "observed_value": 1.0,
            "gate_passed": True,
            "status": "complete_mock",
            "verdict_class": "circular_positive",
            "solver_versions": {},
        },
    )
    relative = mod.build_artifact(
        repo_root=tmp_path,
        result_path=Path("relative.json"),
        work_root=Path("relative-work"),
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )
    assert relative["preconditions_checked"]["result_path"].endswith("relative.json")

    monkeypatch.setattr(mod, "validate_artifact", lambda payload: ["forced validation error"])
    with pytest.raises(ValueError, match="forced validation error"):
        mod.build_artifact(
            repo_root=tmp_path,
            result_path=tmp_path / "bad.json",
            work_root=tmp_path / "bad-work",
            write=False,
            duration_s=1.0,
            tests_run=TESTS_RUN,
            run_date="20260823",
        )


def test_scenario_store_6522_null_blocked_and_disqualified_branches(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-STORE-6522-SEQUENTIAL-EVIDENCE: verdict branches are explicit."""

    aggregate = deepcopy(artifact["aggregate_row_recomputation"])
    gates = deepcopy(artifact["gate_check_summary"])
    aggregate["charged_held_future_benefit_positive"] = False
    aggregate["candidate_score_from_rows"] = 0.0
    assert mod.status_and_verdict(aggregate, gates)[2] is None

    blocked_gates = deepcopy(gates)
    blocked_gates["checks"]["upstream_gate_passed"] = False
    blocked_gates["all_gates_passed"] = False
    assert mod.status_and_verdict(aggregate, blocked_gates)[2] == "blocked"

    unsafe = deepcopy(aggregate)
    unsafe["unsafe_write_count"] = 1
    assert mod.status_and_verdict(unsafe, gates)[2] == "disqualified"

    partial = deepcopy(aggregate)
    partial["execution_complete_score_from_rows"] = 0.0
    assert mod.status_and_verdict(partial, gates)[2] == "partial"
