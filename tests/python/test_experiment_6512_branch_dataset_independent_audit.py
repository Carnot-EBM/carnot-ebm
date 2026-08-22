"""Tests for Exp6512 independent branch-dataset audit.

Spec refs: REQ-BENCH-6512, SCENARIO-BENCH-6512-MISSING-UPSTREAM,
SCENARIO-BENCH-6512-ROW-REPLAY, SCENARIO-BENCH-6512-SPLIT-LINEAGE,
SCENARIO-BENCH-6512-SHARDS-CENSORING, SCENARIO-BENCH-6512-LEAKAGE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_6512_branch_dataset_independent_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6512_branch_dataset_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6512_branch_dataset_independent_audit.py "
    "-m pytest tests/python/test_experiment_6512_branch_dataset_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6512_branch_dataset_independent_audit.py "
    "--fail-under=100 --show-missing"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6512_branch_dataset_independent_audit --date 20260822"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6512_branch_dataset_independent_audit.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6512_branch_dataset_independent_audit.json"
)
EXCLUSION_LINT_COMMAND = ".venv/bin/python scripts/exclusion_manifest_lint.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6512_branch_dataset_independent_audit.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6512_branch_dataset_independent_audit --validate"
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


def _sha(n: int) -> str:
    return "sha256:" + f"{n:064x}"


def _valid_dataset() -> dict[str, Any]:
    rows = [
        {
            "row_id": "u-train",
            "split": "train",
            "base_lineage_id": "base-train",
            "base_instance_hash": _sha(1),
            "exact_label": True,
            "exact_budget": 100,
            "checkpoint_id": "ck-train",
            "shard_id": 0,
            "terminal_disposition": "solved",
            "exact_receipt": {
                "row_id": "u-train",
                "exact_label": True,
                "valid": True,
                "base_instance_hash": _sha(1),
            },
        },
        {
            "row_id": "u-dev",
            "split": "development",
            "base_lineage_id": "base-dev",
            "base_instance_hash": _sha(2),
            "exact_label": False,
            "exact_budget": 100,
            "checkpoint_id": "ck-dev",
            "shard_id": 1,
            "terminal_disposition": "unsat",
            "exact_receipt": {
                "row_id": "u-dev",
                "exact_label": False,
                "valid": True,
                "base_instance_hash": _sha(2),
            },
        },
        {
            "row_id": "u-held",
            "split": "held",
            "base_lineage_id": "base-held",
            "base_instance_hash": _sha(3),
            "exact_label": True,
            "exact_budget": 100,
            "checkpoint_id": "ck-held",
            "shard_id": 1,
            "terminal_disposition": "solved",
            "exact_receipt": {
                "row_id": "u-held",
                "exact_label": True,
                "valid": True,
                "base_instance_hash": _sha(3),
            },
        },
    ]
    return {
        "status": "complete_exact_branch_counterfactual_dataset_v2",
        "verdict_class": "null",
        "branch_counterfactual_rows": rows,
        "shard_manifest": {
            "complete": True,
            "expected_shard_count": 2,
            "terminal_row_count": 3,
            "censored_row_count": 0,
            "hash_chain": [_sha(101), _sha(102)],
            "resume_receipts": [
                {"shard_id": 0, "sha256": _sha(201)},
                {"shard_id": 1, "sha256": _sha(202)},
            ],
            "shards": [
                {"shard_id": 0, "row_count": 1, "sha256": _sha(101)},
                {"shard_id": 1, "row_count": 2, "sha256": _sha(102)},
            ],
        },
        "feature_schema": {
            "features": [
                {"name": "clause_count", "available_at": "decision_time"},
                {"name": "variable_count", "available_at": "static_instance"},
            ]
        },
        "aggregate_row_recomputation": {"row_count": 999},
    }


def _write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_req_bench_6512_spec_declares_audited_gate_contract() -> None:
    """REQ-BENCH-6512: OpenSpec owns the closed readiness gate."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6512") : text.index("REQ-BENCH-3389")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-BENCH-6512-MISSING-UPSTREAM",
        "SCENARIO-BENCH-6512-ROW-REPLAY",
        "SCENARIO-BENCH-6512-SPLIT-LINEAGE",
        "SCENARIO-BENCH-6512-SHARDS-CENSORING",
        "SCENARIO-BENCH-6512-LEAKAGE",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.UPSTREAM_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "branch_dataset_audited_ready_score",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_bench_6512_missing_upstream_closed_artifact(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6512-MISSING-UPSTREAM: absent Exp6511 still closes."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    missing_path = tmp_path / "missing_exp6511.json"

    artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=result_path,
        upstream_path=missing_path,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260822",
    )
    written = json.loads(result_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "blocked_branch_dataset_independent_audit"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["branch_dataset_audited_ready_score"] == 0.0
    assert artifact["upstream_artifact_receipt"]["exists"] is False
    assert artifact["upstream_artifact_receipt"]["sha256"] == "missing"
    assert artifact["upstream_artifact_receipt"]["row_count"] == 0
    assert artifact["exact_receipt_replay_rows"] == []
    assert artifact["per_unit_rows"] == []
    assert artifact["aggregate_row_recomputation"]["row_count"] == 0
    assert artifact["gate_check_summary"][0]["check"] == "upstream_exists"
    assert str(missing_path) in artifact["gate_check_summary"][0]["observed"]
    assert artifact["shortcut_attack_matrix"]["all_attacks_fail_closed"] is True
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_scenario_bench_6512_valid_rows_replay_and_pass(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6512-ROW-REPLAY: valid rows produce a ready score."""

    upstream = tmp_path / "exp6511-valid.json"
    _write_payload(upstream, _valid_dataset())

    artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "exp6512-valid.json",
        upstream_path=upstream,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260822",
    )

    assert artifact["status"] == "complete_branch_dataset_independent_audit_ready"
    assert artifact["verdict_class"] == "null"
    assert artifact["branch_dataset_audited_ready_score"] == 1.0
    assert artifact["upstream_artifact_receipt"]["exists"] is True
    assert artifact["upstream_artifact_receipt"]["terminal_status"] is True
    assert artifact["independent_row_recomputation"]["row_field_used"] == (
        "branch_counterfactual_rows"
    )
    assert artifact["independent_row_recomputation"]["row_count"] == 3
    assert artifact["independent_row_recomputation"]["imported_aggregate_matches"] is False
    assert len(artifact["exact_receipt_replay_rows"]) == 3
    assert all(row["exact_receipt_replay_passed"] for row in artifact["exact_receipt_replay_rows"])
    assert artifact["split_and_lineage_audit"]["sealed_split_passed"] is True
    assert artifact["shard_and_censoring_audit"]["shard_and_censoring_passed"] is True
    assert artifact["feature_timing_audit"]["feature_timing_passed"] is True
    assert artifact["shortcut_attack_matrix"]["all_attacks_fail_closed"] is True
    assert len(artifact["per_unit_rows"]) == 3
    assert artifact["aggregate_row_recomputation"]["split_counts"] == {
        "development": 1,
        "held": 1,
        "train": 1,
    }
    assert artifact["gate_check_summary"] == []
    assert mod.validate_artifact(artifact) == []


def test_scenario_bench_6512_lineage_shard_and_leakage_blocks(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6512-SPLIT-LINEAGE/SHARDS-CENSORING/LEAKAGE: bad rows block."""

    payload = _valid_dataset()
    payload["branch_counterfactual_rows"][1]["base_lineage_id"] = "base-train"
    payload["branch_counterfactual_rows"][1]["checkpoint_id"] = "ck-train"
    payload["branch_counterfactual_rows"][1]["exact_budget"] = 101
    payload["branch_counterfactual_rows"][2]["post_held_repair"] = True
    payload["branch_counterfactual_rows"][2].pop("terminal_disposition")
    payload["branch_counterfactual_rows"][2]["exact_receipt"]["valid"] = False
    payload["branch_counterfactual_rows"].append(
        {
            "row_id": "u-censored",
            "split": "held",
            "base_lineage_id": "base-censored",
            "base_instance_hash": _sha(4),
            "exact_label": False,
            "exact_budget": 100,
            "checkpoint_id": "ck-censored",
            "shard_id": 3,
            "terminal_disposition": "timeout",
            "censored": True,
            "exact_receipt": {
                "row_id": "u-censored",
                "exact_label": True,
                "valid": True,
                "base_instance_hash": _sha(999),
            },
        }
    )
    payload["feature_schema"]["features"].extend(
        [
            {"name": "exact_label", "available_at": "post_decision"},
            {"name": "future_effort", "available_at": "after_solver"},
        ]
    )
    payload["shard_manifest"]["complete"] = False
    payload["shard_manifest"]["expected_shard_count"] = 4
    payload["shard_manifest"]["terminal_row_count"] = 3
    payload["shard_manifest"]["censored_row_count"] = 0
    payload["shard_manifest"]["hash_chain"] = [_sha(101)]
    payload["shard_manifest"]["resume_receipts"] = []
    upstream = tmp_path / "exp6511-invalid.json"
    _write_payload(upstream, payload)

    artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "exp6512-invalid.json",
        upstream_path=upstream,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260822",
    )

    checks = {row["check"]: row["observed"] for row in artifact["gate_check_summary"]}
    assert artifact["status"] == "blocked_branch_dataset_independent_audit"
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["branch_dataset_audited_ready_score"] == 0.0
    assert artifact["independent_row_recomputation"]["exact_receipt_failure_count"] == 2
    assert artifact["split_and_lineage_audit"]["sealed_split_passed"] is False
    assert artifact["split_and_lineage_audit"]["base_lineage_overlap_count"] == 1
    assert artifact["split_and_lineage_audit"]["duplicate_checkpoint_count"] == 1
    assert artifact["split_and_lineage_audit"]["post_held_repair_count"] == 1
    assert artifact["split_and_lineage_audit"]["asymmetric_budget_count"] == 1
    assert artifact["split_and_lineage_audit"]["missing_terminal_disposition_count"] == 1
    assert artifact["shard_and_censoring_audit"]["shard_and_censoring_passed"] is False
    assert artifact["feature_timing_audit"]["feature_timing_passed"] is False
    assert artifact["shortcut_attack_matrix"]["all_attacks_fail_closed"] is False
    assert checks["exact_receipt_replay"] == "2 failures"
    assert "base_lineage_overlap_count=1" in checks["split_and_lineage_audit"]
    assert "missing_shard_count=1" in checks["shard_and_censoring_audit"]
    assert "future_effort" in checks["feature_timing_audit"]
    assert "label" in checks["shortcut_attack_matrix"]
    assert mod.validate_artifact(artifact) == []


def test_scenario_bench_6512_unreadable_and_validation_paths(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6512-MISSING-UPSTREAM: invalid JSON also blocks safely."""

    upstream = tmp_path / "exp6511-bad.json"
    upstream.write_text("{bad", encoding="utf-8")
    artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "exp6512-bad.json",
        upstream_path=upstream,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260822",
    )

    assert artifact["verdict_class"] == "blocked"
    assert artifact["upstream_artifact_receipt"]["json_readable"] is False
    assert artifact["branch_dataset_audited_ready_score"] == 0.0
    assert any(row["check"] == "upstream_json_readable" for row in artifact["gate_check_summary"])

    non_object = tmp_path / "exp6511-list.json"
    non_object.write_text("[]", encoding="utf-8")
    non_object_artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "exp6512-list.json",
        upstream_path=non_object,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260822",
    )
    assert non_object_artifact["upstream_artifact_receipt"]["json_error"] == (
        "top-level JSON is not an object"
    )

    blocked_payload = _valid_dataset()
    blocked_payload["status"] = "blocked_exact_branch_counterfactual_dataset_v2"
    blocked_upstream = tmp_path / "exp6511-blocked.json"
    _write_payload(blocked_upstream, blocked_payload)
    blocked_artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "exp6512-upstream-blocked.json",
        upstream_path=blocked_upstream,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260822",
    )
    assert any(
        row["check"] == "upstream_terminal_status"
        for row in blocked_artifact["gate_check_summary"]
    )

    relative_upstream = tmp_path / "relative-exp6511.json"
    _write_payload(relative_upstream, _valid_dataset())
    relative_artifact = mod.build_artifact(
        repo_root=tmp_path,
        result_path=tmp_path / "relative-result.json",
        upstream_path=Path("relative-exp6511.json"),
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260822",
    )
    assert relative_artifact["upstream_artifact_receipt"]["exists"] is True

    missing_receipt = mod.exact_receipt_replay_rows(
        [{"row_id": "no-receipt", "exact_label": True, "base_instance_hash": _sha(5)}]
    )
    assert missing_receipt[0]["receipt_present"] is False
    assert missing_receipt[0]["exact_receipt_replay_passed"] is False

    mapped_features = mod.feature_timing_audit(
        {"feature_schema": {"features": {"width": {"available_at": "decision_time"}}}}
    )
    string_features = mod.feature_timing_audit({"feature_schema": {"features": ["height"]}})
    bad_features = mod.feature_timing_audit({"feature_schema": {"features": "bad"}})
    assert mapped_features["feature_timing_passed"] is True
    assert string_features["feature_timing_passed"] is True
    assert bad_features["feature_schema_present"] is False

    protected_failure = mod._gate_check_summary(
        artifact["upstream_artifact_receipt"],
        artifact["independent_row_recomputation"],
        artifact["split_and_lineage_audit"],
        artifact["shard_and_censoring_audit"],
        artifact["feature_timing_audit"],
        artifact["shortcut_attack_matrix"],
        {"all_protected_files_unchanged": False, "changed_paths": ["protected.json"]},
    )
    assert any(row["check"] == "protected_files_unchanged" for row in protected_failure)

    validation_mutations = [
        ("required field set mismatch", lambda item: item.pop("status")),
        ("field_principles mismatch", lambda item: item.__setitem__("field_principles", {})),
        ("field_provenance must cover required fields", lambda item: item.__setitem__("field_provenance", {})),
        (
            "branch_dataset_audited_ready_score must be 0.0 or 1.0",
            lambda item: item.__setitem__("branch_dataset_audited_ready_score", 0.5),
        ),
        (
            "score 1.0 requires exact receipts, complete shards, sealed splits, and decision-time features",
            lambda item: item.__setitem__("branch_dataset_audited_ready_score", 1.0),
        ),
        (
            "valid readiness requires verdict_class null",
            lambda item: (
                item.__setitem__("branch_dataset_audited_ready_score", 1.0),
                item.__setitem__("verdict_class", "positive"),
            ),
        ),
        (
            "invalid readiness requires verdict_class blocked or disqualified",
            lambda item: item.__setitem__("verdict_class", "positive"),
        ),
        (
            "inference_substrate mismatch",
            lambda item: item.__setitem__("inference_substrate", "live_llm_inference"),
        ),
        (
            "verifier_is_oracle must be true for label and receipt checks",
            lambda item: item.__setitem__("verifier_is_oracle", False),
        ),
        (
            "score 0.0 requires gate_check_summary entries",
            lambda item: item.__setitem__("gate_check_summary", []),
        ),
        (
            "reproducibility_checksum mismatch",
            lambda item: item.__setitem__("reproducibility_checksum", "sha256:bad"),
        ),
        (
            "protected files changed during audit",
            lambda item: item["protected_files_unchanged"].__setitem__(
                "all_protected_files_unchanged", False
            ),
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


def test_req_bench_6512_main_and_validate_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-BENCH-6512: CLI writes and validates the audit artifact."""

    upstream = tmp_path / "exp6511-valid.json"
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    _write_payload(upstream, _valid_dataset())

    assert (
        mod.main(
            [
                "--date",
                "20260822",
                "--result-path",
                str(result_path),
                "--upstream-path",
                str(upstream),
            ]
        )
        == 0
    )
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert payload["branch_dataset_audited_ready_score"] == 1.0
    assert payload["reproducibility_checksum"] == mod.reproducibility_checksum(payload)

    bad_json = tmp_path / "bad-result.json"
    bad_json.write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="Expecting property name"):
        mod.main(["--validate", "--result-path", str(bad_json)])

    bad_artifact = tmp_path / "bad-artifact.json"
    broken_payload = dict(payload)
    broken_payload.pop("status")
    bad_artifact.write_text(json.dumps(broken_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="required field set mismatch"):
        mod.main(["--validate", "--result-path", str(bad_artifact)])

    monkeypatch.setattr(mod, "validate_artifact", lambda value: ["forced build validation error"])
    with pytest.raises(ValueError, match="forced build validation error"):
        mod.build_artifact(
            repo_root=REPO,
            result_path=tmp_path / "forced-build.json",
            upstream_path=upstream,
            write=False,
            duration_s=1.0,
            tests_run=TESTS_RUN,
            run_date="20260822",
        )

    monkeypatch.undo()
    with pytest.raises(ValueError, match="forced validation error"):
        mod.run(
            date="20260822",
            result_path=tmp_path / "forced.json",
            upstream_path=upstream,
            validate=lambda value: ["forced validation error"],
        )
