"""Tests for Exp6530 external constraint corpus audit.

Spec refs: REQ-BENCH-6530, SCENARIO-BENCH-6530-MISSING,
SCENARIO-BENCH-6530-SOURCE, SCENARIO-BENCH-6530-CHRONOLOGY,
SCENARIO-BENCH-6530-EXACT, SCENARIO-BENCH-6530-SPLIT,
SCENARIO-BENCH-6530-SHARDS, SCENARIO-BENCH-6530-ATTACKS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6530_external_constraint_corpus_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6530_external_constraint_corpus_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6530_external_constraint_corpus_audit.py "
    "-m pytest tests/python/test_experiment_6530_external_constraint_corpus_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6530_external_constraint_corpus_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6530_external_constraint_corpus_audit.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6530_external_constraint_corpus_audit.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6530_external_constraint_corpus_audit.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6530_external_constraint_corpus_audit "
    "--date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6530_external_constraint_corpus_audit --validate"
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


def _source_rows() -> list[dict[str, Any]]:
    return [
        {
            "source_row_id": "src-held-0",
            "domain": "seating",
            "family": "seating",
            "base_problem_id": "base-held",
            "source_problem_id": "base-held",
            "turn_index": 0,
            "event_id": "base-held:0",
            "entities": ["seat_a", "seat_b"],
            "constraints": [{"var": "seat_a", "equals": "red"}],
        },
        {
            "source_row_id": "src-train-0",
            "domain": "scheduling",
            "family": "scheduling",
            "base_problem_id": "base-train",
            "source_problem_id": "base-train",
            "turn_index": 0,
            "event_id": "base-train:0",
            "entities": ["slot_a", "slot_b"],
            "constraints": [
                {"var": "slot_a", "equals": "morning"},
                {"var": "slot_a", "equals": "evening"},
            ],
        },
        {
            "source_row_id": "src-dev-0",
            "domain": "logic_grid",
            "family": "logic_grid",
            "base_problem_id": "base-dev",
            "source_problem_id": "base-dev",
            "turn_index": 0,
            "event_id": "base-dev:0",
            "entities": ["person_a", "person_b"],
            "constraints": [{"var": "person_a", "not_equals": "blue"}],
        },
    ]


def _fixture_rows(source_hash: str, source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    splits = ["held", "train", "development"]
    labels = ["satisfiable", "contradiction", "satisfiable"]
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows):
        rows.append(
            {
                "local_unit_id": f"unit-{splits[index]}-0",
                "source_row_id": source["source_row_id"],
                "source_file_relpath": "data/problems/sample.jsonl",
                "source_file_sha256": source_hash,
                "source_row_hash": mod.sha256_json(source),
                "domain": source["domain"],
                "family": source["family"],
                "base_problem_id": source["base_problem_id"],
                "source_problem_id": source["source_problem_id"],
                "split_name": splits[index],
                "turn_index": source["turn_index"],
                "chronology_index": index,
                "event_id": source["event_id"],
                "terminal_disposition": "terminal",
                "exact_label": labels[index],
                "contradiction": labels[index] == "contradiction",
                "drift": labels[index] == "satisfiable",
                "hardness_bin": "easy" if index != 1 else "hard",
                "censored": False,
                "raw_turns": [source],
                "constraints": source["constraints"],
                "answer": {"label": labels[index]},
                "solver_receipt": {"cached": True, "ignored_by_exp6530": True},
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_fixture_repo(tmp_path: Path, *, tamper_label: bool = False) -> Path:
    repo = tmp_path / "repo"
    source_root = repo / "upstream" / "drift-bench"
    source_file = source_root / "data" / "problems" / "sample.jsonl"
    source_rows = _source_rows()
    _write_jsonl(source_file, source_rows)
    (source_root / "README.md").write_text(
        "DRIFT-Bench note: original SQLite run databases were corrupted.",
        encoding="utf-8",
    )
    (source_root / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    (source_root / "data" / "problems" / "README.md").write_text(
        "Problem schema has domain, problem_id, turns, constraints, and answer.",
        encoding="utf-8",
    )
    for protected in [
        "CODEX.md",
        "CLAUDE.md",
        "research-roadmap.yaml",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "results/experiment_6514_atomic_shard_artifact_transaction.json",
    ]:
        path = repo / protected
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"protected {protected}\n", encoding="utf-8")

    source_hash = mod.sha256_file(source_file)
    fixture_rows = _fixture_rows(source_hash, source_rows)
    if tamper_label:
        fixture_rows[1]["exact_label"] = "satisfiable"
    fixture_path = repo / mod.FIXTURE_RELATIVE_PATH
    _write_jsonl(fixture_path, fixture_rows)
    fixture_hash = mod.sha256_file(fixture_path)

    shard_path = repo / "results" / "fixtures" / "v565_drift_shards" / "shard-000.jsonl"
    _write_jsonl(shard_path, fixture_rows)
    shard_hash = mod.sha256_file(shard_path)
    planned_ids = [row["local_unit_id"] for row in fixture_rows]
    exact_counts = {"contradiction": 1, "satisfiable": 2}
    if tamper_label:
        exact_counts = {"satisfiable": 3}
    intake = {
        "status": "complete_drift_bench_external_intake_ready",
        "honest_verdict": "complete_drift_bench_external_intake_ready",
        "source_revision_and_license_receipt": {
            "repo_url": "https://github.com/kaons-research/drift-bench",
            "immutable_revision": "a" * 40,
            "source_root": "upstream/drift-bench",
            "license": "MIT",
            "license_verified": True,
            "data_schema_path": "data/problems/README.md",
            "data_schema_verified": True,
            "upstream_corruption_warning_present": True,
        },
        "source_file_hashes": {"data/problems/sample.jsonl": source_hash},
        "shard_manifest": {
            "planned_unit_ids": planned_ids,
            "terminal_unit_ids": planned_ids,
            "journal_chain": [{"index": 0, "sha256": shard_hash}],
            "resume_receipts": [{"resume_id": "resume-0", "verified": True}],
            "shards": [
                {
                    "path": "results/fixtures/v565_drift_shards/shard-000.jsonl",
                    "sha256": shard_hash,
                    "row_count": len(fixture_rows),
                }
            ],
            "final_atomic_write_receipt": {
                "final_path": mod.FIXTURE_RELATIVE_PATH.as_posix(),
                "final_sha256": fixture_hash,
                "row_count": len(fixture_rows),
            },
        },
        "planned_and_terminal_unit_counts": {
            "planned_count": len(fixture_rows),
            "terminal_count": len(fixture_rows),
        },
        "fixture_path_and_hash": {
            "path": mod.FIXTURE_RELATIVE_PATH.as_posix(),
            "sha256": fixture_hash,
            "row_count": len(fixture_rows),
        },
        "aggregate_row_recomputation": {
            "row_count": len(fixture_rows),
            "split_counts": {"development": 1, "held": 1, "train": 1},
            "exact_label_counts": exact_counts,
        },
        "external_constraint_corpus_ready_score": 1.0,
    }
    intake_path = repo / mod.EXP6529_RELATIVE_PATH
    intake_path.parent.mkdir(parents=True, exist_ok=True)
    intake_path.write_text(json.dumps(intake, indent=2, sort_keys=True), encoding="utf-8")
    return repo


def _artifact(repo: Path, result_path: Path) -> dict[str, Any]:
    return mod.build_artifact(
        repo_root=repo,
        result_path=result_path,
        run_date="20260823",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        now_utc="2026-08-23T12:00:00Z",
    )


def test_req_bench_6530_spec_declares_audit_contract() -> None:
    """REQ-BENCH-6530: OpenSpec owns the Exp6530 audit contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6530") :]

    for token in (
        "SCENARIO-BENCH-6530-MISSING",
        "SCENARIO-BENCH-6530-SOURCE",
        "SCENARIO-BENCH-6530-CHRONOLOGY",
        "SCENARIO-BENCH-6530-EXACT",
        "SCENARIO-BENCH-6530-SPLIT",
        "SCENARIO-BENCH-6530-SHARDS",
        "SCENARIO-BENCH-6530-ATTACKS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "external_constraint_corpus_audited_ready_score",
        mod.INFERENCE_SUBSTRATE,
        "`verifier_is_oracle` SHALL be true",
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_bench_6530_missing_writes_blocked_artifact(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6530-MISSING: missing Exp6529 inputs still close the artifact."""

    repo = tmp_path / "repo"
    repo.mkdir()
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = _artifact(repo, result_path)
    written = json.loads(result_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "blocked_external_constraint_corpus_audit"
    assert artifact["honest_verdict"].startswith("blocked_external_constraint_corpus_audit")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["external_constraint_corpus_audited_ready_score"] == 0.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["gate_check_summary"]["all_gates_passed"] is False
    failures = artifact["gate_check_summary"]["failed_checks"]
    assert {
        (row["check"], row["observed"])
        for row in failures
        if row["check"] in {"exp6529_artifact_exists", "fixture_exists"}
    } == {("exp6529_artifact_exists", False), ("fixture_exists", False)}
    assert artifact["source_existence_and_hash_receipts"]["intake_artifact"]["exists"] is False
    assert artifact["source_existence_and_hash_receipts"]["fixture"]["exists"] is False
    assert artifact["aggregate_row_recomputation"]["ready_score_from_rows"] == 0.0
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)


def test_scenarios_bench_6530_complete_audit_recomputes_all_rows(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6530-SOURCE/CHRONOLOGY/EXACT/SPLIT/SHARDS: rows are replayed."""

    repo = _write_fixture_repo(tmp_path)
    artifact = _artifact(repo, tmp_path / mod.RESULT_RELATIVE_PATH.name)

    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_external_constraint_corpus_audit"
    assert artifact["honest_verdict"].startswith("complete_external_constraint_corpus_audit")
    assert artifact["verdict_class"] is None
    assert artifact["external_constraint_corpus_audited_ready_score"] == 1.0
    assert artifact["gate_check_summary"]["all_gates_passed"] is True

    source_receipt = artifact["independent_revision_and_license_receipt"]
    assert source_receipt["immutable_revision"] == "a" * 40
    assert source_receipt["license_verified"] is True
    assert source_receipt["corruption_boundary_text_verified"] is True
    assert source_receipt["schema_verified"] is True
    assert all(row["passed"] is True for row in artifact["source_identity_audit_rows"])
    assert all(row["chronology_valid"] is True for row in artifact["chronology_replay_rows"])
    assert all(
        row["replayed_label_matches"] is True for row in artifact["independent_exact_replay_rows"]
    )
    assert artifact["split_and_lineage_audit"]["base_problem_overlap_count"] == 0
    assert artifact["split_and_lineage_audit"]["missing_terminal_disposition_count"] == 0
    assert artifact["shard_and_transaction_audit"]["fixture_hash_matches"] is True
    assert artifact["shard_and_transaction_audit"]["row_count_matches"] is True

    aggregate = artifact["aggregate_row_recomputation"]
    assert aggregate["fixture_row_count"] == 3
    assert aggregate["split_counts"] == {"development": 1, "held": 1, "train": 1}
    assert aggregate["exact_label_counts"] == {"contradiction": 1, "satisfiable": 2}
    assert aggregate["ready_score_from_rows"] == 1.0
    assert {row["attack"] for row in artifact["leakage_attack_matrix"]} >= {
        "aggregate_inheritance_attack",
        "source_id_trust_attack",
        "solver_cache_trust_attack",
    }
    assert all(row["passed"] is True for row in artifact["leakage_attack_matrix"])
    assert {row["row_type"] for row in artifact["per_unit_rows"]} >= {
        "source_identity",
        "chronology",
        "exact_replay",
        "aggregate",
        "leakage_attack",
    }


def test_scenario_bench_6530_tampering_disqualifies_without_trusting_aggregates(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6530-ATTACKS: cached labels and inherited aggregates fail closed."""

    repo = _write_fixture_repo(tmp_path, tamper_label=True)
    artifact = _artifact(repo, tmp_path / mod.RESULT_RELATIVE_PATH.name)
    failures = artifact["gate_check_summary"]["failed_checks"]

    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "disqualified_external_constraint_corpus_audit"
    assert artifact["honest_verdict"].startswith("disqualified_external_constraint_corpus_audit")
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["external_constraint_corpus_audited_ready_score"] == 0.0
    assert any(row["check"] == "exact_label_replay" for row in failures)
    assert any(row["check"] == "aggregate_tampering" for row in failures)
    replay_rows = {row["local_unit_id"]: row for row in artifact["independent_exact_replay_rows"]}
    assert replay_rows["unit-train-0"]["observed_label"] == "satisfiable"
    assert replay_rows["unit-train-0"]["recomputed_label"] == "contradiction"
    aggregate_attack = [
        row
        for row in artifact["leakage_attack_matrix"]
        if row["attack"] == "aggregate_inheritance_attack"
    ][0]
    assert aggregate_attack["passed"] is False
    assert aggregate_attack["observed"] == {
        "row_count": 3,
        "split_counts": {"development": 1, "held": 1, "train": 1},
        "exact_label_counts": {"contradiction": 1, "satisfiable": 2},
    }


def test_scenario_bench_6530_validation_cli_and_edge_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-BENCH-6530-ATTACKS: defensive validators fail closed."""

    repo = _write_fixture_repo(tmp_path)
    result_path = tmp_path / "edge-artifact.json"
    artifact = _artifact(repo, result_path)

    assert mod._utc_now().endswith("Z")
    assert mod.sha256_file(tmp_path / "missing") == "missing"
    assert "git_available" in mod._git_state(REPO)
    assert mod._constraints_from_source({"source_row_hash": "missing"}, {}) == []
    assert mod._observed_label({}) == "missing"
    assert (
        mod._constraint_label(
            [
                {"variable": "x", "op": "=", "value": "a"},
                {"var": "x", "op": "==", "value": "b"},
            ]
        )
        == "contradiction"
    )
    assert (
        mod._constraint_label(
            [
                {"var": "", "equals": "ignored"},
                {"var": "y", "equals": "red"},
                {"var": "y", "not_equals": "red"},
            ]
        )
        == "contradiction"
    )
    assert mod._constraint_label([{"var": "z", "op": "!=", "value": "blue"}]) == "satisfiable"
    assert mod._status_for_class("partial") == "partial_external_constraint_corpus_audit"

    duplicate_rows = [
        {
            "local_unit_id": "dup-0",
            "base_problem_id": "base",
            "split_name": "held",
            "family": "Logic Grid",
            "event_id": "event",
            "terminal_disposition": "terminal",
        },
        {
            "local_unit_id": "dup-1",
            "base_problem_id": "base",
            "split_name": "train",
            "family": "logic-grid",
            "event_id": "event",
            "post_held_repair": True,
        },
    ]
    split = mod.split_and_lineage_audit(
        duplicate_rows,
        [{"chronology_gap": True, "duplicate_event": True}],
    )
    assert split["base_problem_overlap_count"] == 1
    assert split["duplicate_event_count"] == 2
    assert split["chronology_gap_count"] == 1
    assert split["family_alias_collision_count"] == 1
    assert split["post_held_repair_count"] == 1
    assert split["missing_terminal_disposition_count"] == 1
    assert split["passed"] is False

    skipped_shard = mod.shard_and_transaction_audit(
        repo_root=repo,
        intake={"shard_manifest": {"shards": ["bad"]}},
        fixture_path=repo / mod.FIXTURE_RELATIVE_PATH,
        fixture_rows=[],
    )
    assert skipped_shard["shard_rows"] == []
    assert skipped_shard["passed"] is False

    gate = mod.gate_check_summary(
        existence_receipts={
            "intake_artifact": {"exists": True},
            "fixture": {"exists": True},
        },
        revision_receipt={
            "source_root_exists": True,
            "revision_is_immutable": True,
            "license_verified": True,
            "schema_verified": True,
            "corruption_boundary_text_verified": True,
        },
        source_identity_rows=[{"passed": False}],
        chronology_rows=[{"chronology_valid": False}],
        exact_rows=[],
        split_audit={"passed": False},
        shard_audit={"passed": False},
        aggregate_rows=[],
        attack_rows=[],
    )
    assert {row["check"] for row in gate["failed_checks"]} >= {
        "source_identity_hash",
        "chronology_replay",
        "split_lineage",
        "shard_transaction",
    }

    bad = json.loads(json.dumps(artifact))
    bad.pop("status")
    bad["field_principles"] = {}
    bad["field_provenance"] = {}
    bad["verdict_class"] = "positive"
    bad["honest_verdict"] = "ready"
    bad["inference_substrate"] = "live_llm"
    bad["verifier_is_oracle"] = False
    bad["external_constraint_corpus_audited_ready_score"] = 0.5
    bad["reproducibility_checksum"] = "bad"
    errors = mod.validate_artifact(bad)
    assert "required field set mismatch" in errors
    assert "field_principles mismatch" in errors
    assert "field_provenance must cover required fields" in errors
    assert "verdict_class outside Exp6530 enum" in errors
    assert "honest_verdict terminal prefix mismatch" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "ready score mismatch" in errors
    assert "reproducibility_checksum mismatch" in errors

    bad_ready = json.loads(json.dumps(artifact))
    bad_ready["gate_check_summary"] = {"all_gates_passed": False, "failed_checks": []}
    bad_ready["reproducibility_checksum"] = mod.reproducibility_checksum(bad_ready)
    assert "ready artifact must have all gates passed" in mod.validate_artifact(bad_ready)

    bad_blocked = json.loads(json.dumps(artifact))
    bad_blocked["external_constraint_corpus_audited_ready_score"] = 0.0
    bad_blocked["aggregate_row_recomputation"]["ready_score_from_rows"] = 0.0
    bad_blocked["gate_check_summary"] = {"all_gates_passed": True, "failed_checks": []}
    bad_blocked["reproducibility_checksum"] = mod.reproducibility_checksum(bad_blocked)
    assert "blocked artifact cannot have all gates passed" in mod.validate_artifact(bad_blocked)

    monkeypatch.setattr(mod, "REPO_ROOT", repo)
    cli_path = tmp_path / "cli-artifact.json"
    assert mod.main(["--date", "20260823", "--result-path", str(cli_path)]) == 0
    assert f"wrote {mod.RESULT_RELATIVE_PATH.as_posix()}" in capsys.readouterr().out
    assert mod.main(["--result-path", str(cli_path), "--validate"]) == 0
    assert f"validated {mod.RESULT_RELATIVE_PATH.as_posix()}" in capsys.readouterr().out
    bad_cli_path = tmp_path / "bad-cli.json"
    bad_cli_path.write_text("{}", encoding="utf-8")
    assert mod.main(["--result-path", str(bad_cli_path), "--validate"]) == 1
    assert "required field set mismatch" in capsys.readouterr().out

    monkeypatch.setattr(mod, "build_artifact", lambda **kwargs: {"bad": True})
    monkeypatch.setattr(mod, "validate_artifact", lambda artifact: ["forced error"])
    assert mod.main(["--result-path", str(tmp_path / "forced.json")]) == 1
    assert "forced error" in capsys.readouterr().out
