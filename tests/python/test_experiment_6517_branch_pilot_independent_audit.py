"""Tests for Exp6517 branch pilot independent audit.

Spec refs: REQ-BENCH-6517, SCENARIO-BENCH-6517-MISSING-UPSTREAM,
SCENARIO-BENCH-6517-ROW-REPLAY, SCENARIO-BENCH-6517-SHARDS,
SCENARIO-BENCH-6517-SPLIT-TIMING, SCENARIO-BENCH-6517-ATTACKS,
SCENARIO-BENCH-6517-TERMINAL.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6517_branch_pilot_independent_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6517_branch_pilot_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6517_branch_pilot_independent_audit.py "
    "-m pytest tests/python/test_experiment_6517_branch_pilot_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6517_branch_pilot_independent_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6517_branch_pilot_independent_audit.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6517_branch_pilot_independent_audit.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6517_branch_pilot_independent_audit.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6517_branch_pilot_independent_audit --validate"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6517_branch_pilot_independent_audit --date 20260823"
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
def valid_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-BENCH-6517: build a temp audit without touching tracked results."""

    root = tmp_path_factory.mktemp("exp6517")
    return mod.build_artifact(
        repo_root=REPO,
        result_path=root / mod.RESULT_RELATIVE_PATH.name,
        source_path=REPO / mod.UPSTREAM_RELATIVE_PATH,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )


def test_req_bench_6517_spec_declares_audit_contract() -> None:
    """REQ-BENCH-6517: OpenSpec owns the independent audit contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6517") : text.index("REQ-BENCH-3389")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-BENCH-6517-MISSING-UPSTREAM",
        "SCENARIO-BENCH-6517-ROW-REPLAY",
        "SCENARIO-BENCH-6517-SHARDS",
        "SCENARIO-BENCH-6517-SPLIT-TIMING",
        "SCENARIO-BENCH-6517-ATTACKS",
        "SCENARIO-BENCH-6517-TERMINAL",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "branch_pilot_audited_ready_score",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_bench_6517_terminal_valid_readiness(
    valid_artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6517-TERMINAL: valid readiness is complete."""

    assert set(valid_artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert valid_artifact["status"] == "complete_branch_pilot_independent_audit_ready"
    assert valid_artifact["honest_verdict"].startswith("complete_")
    assert valid_artifact["verdict_class"] is None
    assert valid_artifact["branch_pilot_audited_ready_score"] == 1.0
    assert valid_artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert valid_artifact["verifier_is_oracle"] is True
    assert valid_artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(valid_artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert valid_artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert valid_artifact["reproducibility_checksum"] == mod.reproducibility_checksum(
        valid_artifact
    )
    assert mod.validate_artifact(valid_artifact) == []

    receipt = valid_artifact["upstream_artifact_receipt"]
    assert receipt["exists"] is True
    assert receipt["parse_status"] == "parsed"
    assert receipt["source_status"] == "complete_exact_branch_pilot_dataset_v3_ready"
    assert receipt["source_verdict_class"] is None
    assert receipt["branch_row_count"] == 36
    assert receipt["per_unit_row_count"] == 82
    assert receipt["solver_availability"]["exact_replay_available"] is True
    assert receipt["protected_file_hashes_before"][mod.UPSTREAM_RELATIVE_PATH.as_posix()].startswith(
        "sha256:"
    )


def test_scenario_bench_6517_row_replay_counts_and_receipts(
    valid_artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6517-ROW-REPLAY: rows and exact receipts own readiness."""

    replay_rows = valid_artifact["exact_receipt_replay_rows"]
    recompute = valid_artifact["independent_row_recomputation"]
    aggregate = valid_artifact["aggregate_row_recomputation"]

    assert len(replay_rows) == 36
    assert recompute["branch_row_count"] == 36
    assert recompute["checkpoint_count"] == 18
    assert recompute["candidate_completeness_passed"] is True
    assert recompute["exact_receipt_replay_failure_count"] == 0
    assert recompute["imported_source_ready_score"] == 1.0
    assert recompute["recomputed_ready_score_from_rows"] == 1.0
    assert aggregate["ready_score_from_audit_rows"] == 1.0
    assert aggregate["source_unit_audit_row_count"] == 36

    assert all(row["replayed_exact_status_matches_receipt"] is True for row in replay_rows)
    assert all(row["receipt_unit_id_matches_row"] is True for row in replay_rows)
    assert all(row["base_hash_found_in_exp6504"] is True for row in replay_rows)
    assert all(row["row_hash_recomputed"] is True for row in replay_rows)
    assert all(row["model_or_proof_revalidated"] is True for row in replay_rows)
    assert all(row["audit_passed"] is True for row in replay_rows)
    assert {row["candidate_value"] for row in replay_rows} == {False, True}


def test_scenario_bench_6517_shards_splits_features_censoring_and_attacks(
    valid_artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6517-SHARDS/SPLIT-TIMING/ATTACKS: audits pass."""

    split = valid_artifact["split_and_lineage_audit"]
    transaction = valid_artifact["transaction_and_shard_audit"]
    features = valid_artifact["feature_timing_audit"]
    censoring = valid_artifact["censoring_audit"]
    attacks = valid_artifact["shortcut_attack_matrix"]

    assert split["split_audit_passed"] is True
    assert split["base_lineage_overlap_count"] == 0
    assert split["duplicate_checkpoint_count"] == 0
    assert split["post_held_repair_count"] == 0
    assert split["minimum_cell_floor_observed"] >= split["minimum_cell_floor_required"]
    assert set(split["lineage_sets"]) == {"development", "held", "train"}

    assert transaction["transaction_audit_passed"] is True
    assert transaction["exp6514_ready_score"] == 1.0
    assert transaction["planned_ids_match_branch_rows"] is True
    assert transaction["terminal_ids_match_branch_rows"] is True
    assert transaction["terminal_shard_hashes_match_rows"] is True
    assert transaction["resume_receipts_match_terminal_rows"] is True
    assert transaction["journal_chain_length_matches_rows"] is True
    assert transaction["corrupt_resume_detected"] is True
    assert transaction["final_atomic_receipt_passed"] is True

    assert features["feature_timing_passed"] is True
    assert features["forbidden_features_observed"] == []
    assert features["feature_event_before_replay"] is True
    assert features["checkpoint_selection_uses_outcome"] is False
    assert features["checkpoint_selection_uses_future_effort"] is False

    assert censoring["censoring_audit_passed"] is True
    assert censoring["asymmetric_budget_count"] == 0
    assert censoring["missing_terminal_disposition_count"] == 0
    assert censoring["censored_row_count"] == 0
    assert censoring["timeout_count"] == 0

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_attacks_fail_closed"] is True
    assert attacks["false_accept_count"] == 0
    assert all(row["fail_closed"] is True for row in attacks["rows"])


def test_scenario_bench_6517_missing_and_corrupt_upstreams_close_blocked(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6517-MISSING-UPSTREAM: bad sources still close."""

    missing = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "missing.json",
        source_path=tmp_path / "absent-source.json",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
    )
    assert missing["status"] == "blocked_branch_pilot_independent_audit"
    assert missing["honest_verdict"].startswith("blocked_")
    assert missing["verdict_class"] == "blocked"
    assert missing["branch_pilot_audited_ready_score"] == 0.0
    assert missing["upstream_artifact_receipt"]["exists"] is False
    assert missing["upstream_artifact_receipt"]["parse_status"] == "missing"
    assert "source_available_and_parsed" in missing["gate_check_summary"]["failed_checks"]
    assert json.loads((tmp_path / "missing.json").read_text(encoding="utf-8")) == missing
    assert mod.validate_artifact(missing) == []

    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{not valid json", encoding="utf-8")
    corrupt = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "corrupt-out.json",
        source_path=corrupt_path,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
    )
    assert corrupt["verdict_class"] == "blocked"
    assert corrupt["branch_pilot_audited_ready_score"] == 0.0
    assert corrupt["upstream_artifact_receipt"]["exists"] is True
    assert corrupt["upstream_artifact_receipt"]["parse_status"] == "corrupt_json"
    assert "source_available_and_parsed" in corrupt["gate_check_summary"]["failed_checks"]


@pytest.mark.parametrize(
    ("name", "mutate", "expected_class", "expected_check"),
    [
        (
            "blocked",
            lambda item: (
                item.__setitem__("status", "blocked_exact_branch_pilot_dataset_v3"),
                item.__setitem__("verdict_class", "blocked"),
                item.__setitem__("branch_pilot_dataset_ready_score", 0.0),
            ),
            "blocked",
            "source_declares_complete_ready",
        ),
        (
            "partial",
            lambda item: item.__setitem__("verdict_class", "partial"),
            "blocked",
            "source_declares_complete_ready",
        ),
        (
            "false_receipt",
            lambda item: item["branch_counterfactual_rows"][0]["exact_receipt"].__setitem__(
                "valid", False
            ),
            "disqualified",
            "exact_receipts_replay",
        ),
        (
            "split_leak",
            lambda item: item["branch_counterfactual_rows"][0].__setitem__(
                "split", "held"
            ),
            "disqualified",
            "split_and_lineage",
        ),
        (
            "feature_leak",
            lambda item: item["structural_feature_schema"]["features"].append(
                {"name": "future_effort", "available_at": "post_replay"}
            ),
            "disqualified",
            "feature_timing",
        ),
        (
            "censored",
            lambda item: item["branch_counterfactual_rows"][0].__setitem__("censored", True),
            "disqualified",
            "censoring_and_budgets",
        ),
        (
            "asymmetric_budget",
            lambda item: item["branch_counterfactual_rows"][0].__setitem__("exact_budget", 1),
            "disqualified",
            "censoring_and_budgets",
        ),
        (
            "missing_terminal",
            lambda item: item["branch_counterfactual_rows"][0].__setitem__(
                "terminal_disposition", ""
            ),
            "disqualified",
            "censoring_and_budgets",
        ),
        (
            "missing_candidate",
            lambda item: item["branch_counterfactual_rows"].pop(0),
            "disqualified",
            "censoring_and_budgets",
        ),
        (
            "aggregate_tamper",
            lambda item: item["aggregate_row_recomputation"].__setitem__(
                "ready_score_from_rows", 0.0
            ),
            "disqualified",
            "aggregate_tampering",
        ),
        (
            "one_cell_headroom",
            lambda item: item["split_commitment"].__setitem__("minimum_cell_floor_observed", 0),
            "disqualified",
            "split_and_lineage",
        ),
        (
            "shard_mismatch",
            lambda item: item["shard_manifest"]["resume_receipts"][0].__setitem__(
                "shard_hash", "sha256:" + "0" * 64
            ),
            "disqualified",
            "transaction_and_shards",
        ),
    ],
)
def test_scenario_bench_6517_failure_paths_name_observed_values(
    tmp_path: Path,
    name: str,
    mutate: Any,
    expected_class: str,
    expected_check: str,
) -> None:
    """SCENARIO-BENCH-6517-ATTACKS: invalid upstream paths fail closed."""

    source = deepcopy(json.loads((REPO / mod.UPSTREAM_RELATIVE_PATH).read_text(encoding="utf-8")))
    mutate(source)
    source_path = tmp_path / f"{name}.json"
    source_path.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / f"{name}-audit.json",
        source_path=source_path,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
    )

    assert artifact["verdict_class"] == expected_class
    assert artifact["branch_pilot_audited_ready_score"] == 0.0
    assert expected_check in artifact["gate_check_summary"]["failed_checks"]
    observed = artifact["gate_check_summary"]["checks"][expected_check]["observed"]
    assert observed != artifact["gate_check_summary"]["checks"][expected_check]["expected"]
    assert mod.validate_artifact(artifact) == []


def test_scenario_bench_6517_validation_fails_closed(
    valid_artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6517-TERMINAL: schema validation rejects drift."""

    mutations = [
        ("required field set mismatch", lambda item: item.pop("status")),
        ("field_principles mismatch", lambda item: item.__setitem__("field_principles", {})),
        ("field_provenance must cover required fields", lambda item: item.__setitem__("field_provenance", {})),
        ("verdict_class cannot be positive", lambda item: item.__setitem__("verdict_class", "positive")),
        ("verdict_class outside Exp6517 enum", lambda item: item.__setitem__("verdict_class", "partial")),
        ("inference_substrate mismatch", lambda item: item.__setitem__("inference_substrate", "live_llm")),
        ("verifier_is_oracle must be true", lambda item: item.__setitem__("verifier_is_oracle", False)),
        (
            "branch_pilot_audited_ready_score must be 0.0 or 1.0",
            lambda item: item.__setitem__("branch_pilot_audited_ready_score", 0.5),
        ),
        (
            "ready score mismatch",
            lambda item: item.__setitem__("branch_pilot_audited_ready_score", 0.0),
        ),
        (
            "gate_check_summary mismatch",
            lambda item: item["gate_check_summary"].__setitem__("failed_checks", ["bad"]),
        ),
        (
            "aggregate_row_recomputation mismatch",
            lambda item: item["aggregate_row_recomputation"].__setitem__(
                "ready_score_from_audit_rows", 0.0
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
        (
            "honest_verdict lacks terminal prefix",
            lambda item: item.__setitem__("honest_verdict", "running"),
        ),
    ]

    for expected, mutate in mutations:
        broken = deepcopy(valid_artifact)
        mutate(broken)
        assert expected in mod.validate_artifact(broken)


def test_scenario_bench_6517_malformed_helpers_and_internal_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-BENCH-6517-MISSING-UPSTREAM: malformed helper paths close."""

    non_object = tmp_path / "non-object.json"
    non_object.write_text("[]", encoding="utf-8")
    assert mod._read_json_with_status(non_object)[1] == "non_object"
    assert mod._read_json_object(tmp_path / "missing.json")["_read_status"] == "missing"
    assert mod._source_shard_paths({"shard_manifest": []}) == []
    assert mod.prior_failure_receipt(REPO, {"prior_failure_receipts": []})[
        "source_prior_exp6510_status"
    ] is None
    assert mod.exact_receipt_replay_rows(REPO, {"branch_counterfactual_rows": {}}) == []
    assert mod._source_aggregate_matches({"aggregate_row_recomputation": []}, [], []) is False

    bad_repo = tmp_path / "bad-repo"
    bad_exp6504 = bad_repo / mod.EXP6504_RELATIVE_PATH
    bad_exp6504.parent.mkdir(parents=True)
    bad_exp6504.write_text('{"raw_instance_rows": {}}', encoding="utf-8")
    assert mod._base_rows_by_hash(bad_repo) == {}

    relative = mod.build_artifact(
        repo_root=REPO,
        result_path=Path("results/not-written-exp6517.json"),
        source_path=mod.UPSTREAM_RELATIVE_PATH,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
    )
    assert relative["branch_pilot_audited_ready_score"] == 1.0

    bad_result = tmp_path / "bad-audit.json"
    bad_result.write_text('{"status": "running"}', encoding="utf-8")
    with pytest.raises(ValueError, match="required field set mismatch"):
        mod.main(["--result-path", str(bad_result), "--validate"])

    monkeypatch.setattr(mod, "validate_artifact", lambda _payload: ["forced invalid"])
    with pytest.raises(ValueError, match="forced invalid"):
        mod.build_artifact(
            repo_root=REPO,
            result_path=tmp_path / "forced-invalid.json",
            source_path=REPO / mod.UPSTREAM_RELATIVE_PATH,
            write=False,
            duration_s=1.0,
            tests_run=TESTS_RUN,
        )


def test_scenario_bench_6517_main_and_validate_roundtrip(tmp_path: Path) -> None:
    """REQ-BENCH-6517: CLI writes and validates the audit artifact."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name

    assert (
        mod.main(
            [
                "--date",
                "20260823",
                "--source-path",
                str(REPO / mod.UPSTREAM_RELATIVE_PATH),
                "--result-path",
                str(result_path),
            ]
        )
        == 0
    )
    assert result_path.is_file()
    written = json.loads(result_path.read_text(encoding="utf-8"))
    assert written["branch_pilot_audited_ready_score"] == 1.0
    assert mod.main(["--result-path", str(result_path), "--validate"]) == 0
