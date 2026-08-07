"""Exp6184 task-scoped V536 evidence-isolation preflight tests.

Spec refs:
  REQ-REPORT-6184
  SCENARIO-REPORT-6184-TASK-SCOPE-NON-CLOSURE
  SCENARIO-REPORT-6184-FROZEN-PREFLIGHT-INVOCATION
  SCENARIO-REPORT-6184-COMPATIBLE-WRITERS
  SCENARIO-REPORT-6184-INTERCEPTED-VS-MUTATION
  SCENARIO-REPORT-6184-ESCAPE-ROOT-ATOMIC-QUARANTINE
  SCENARIO-REPORT-6184-SCHEMA-READINESS
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from carnot import experiment_6184_v536_evidence_isolation_preflight as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_current_precondition_snapshot(path: Path) -> Path:
    snapshot = mod.snapshot_repository(REPO)
    payload = {
        "present": True,
        "path": str(path),
        "git_status_short": [],
        "tracked_results_count": snapshot["tracked_results"]["count"],
        "tracked_results_aggregate_sha256": snapshot["tracked_results"]["sha256"],
        "sentinel_result_hashes": {
            key: value["sha256"] for key, value in snapshot["sentinel_hashes"].items()
        },
        "quarantine_fields": snapshot["quarantine_fields"],
        "protected_file_hashes": {
            key: value["sha256"] for key, value in snapshot["protected_files"].items()
        },
        "completion_history_multiplicity": {
            "milestone_2026_08_535_occurrences": 0,
            "milestone_2026_08_536_occurrences": 0,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_req_report_6184_spec_declares_intercepted_attempt_contract() -> None:
    """REQ-REPORT-6184: OpenSpec names the V536 preflight contract."""
    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6184") :]
    section = section[: section.index("## Implementation Status (REQ-REPORT-6184)")]
    for required in [
        "Exp6183 through Exp6196",
        "task-owned temporary artifact root before pytest",
        "expected_intercepted_attempt",
        "actual_mutation",
        "SHALL NOT increment",
        "repository_wide_closure_claimed` SHALL be the bare boolean `false",
        "canonical artifact resolver",
        "legacy literal",
        "checkpoint/resume",
        "subprocess",
        "traversal escape",
        "symlink escape",
        "workspace/repository/results-root rejection",
        "quarantine",
        "deterministic_task_scoped_repository_test_isolation",
        "SCENARIO-REPORT-6184-TASK-SCOPE-NON-CLOSURE",
        "SCENARIO-REPORT-6184-FROZEN-PREFLIGHT-INVOCATION",
        "SCENARIO-REPORT-6184-COMPATIBLE-WRITERS",
        "SCENARIO-REPORT-6184-INTERCEPTED-VS-MUTATION",
        "SCENARIO-REPORT-6184-ESCAPE-ROOT-ATOMIC-QUARANTINE",
        "SCENARIO-REPORT-6184-SCHEMA-READINESS",
    ]:
        assert required in section


def test_scenario_report_6184_task_census_is_bounded_to_declared_v536_tasks(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6184-TASK-SCOPE-NON-CLOSURE: census is not repo-wide."""
    census = mod.collect_v536_task_writer_census(REPO)

    assert census["scope"]["first_task"] == "exp6183-v536-transition"
    assert census["scope"]["last_task"] == "exp6196-v536-capstone"
    assert census["scope"]["repository_wide_writer_scan"] is False
    assert census["declared_task_count"] == 14
    assert {row["task_id"] for row in census["rows"]} == set(mod.V536_TASK_IDS)
    assert all(row["exact_path"].startswith("results/experiment_") for row in census["rows"])
    assert all("scripts/research_conductor.py" not in row["module"] for row in census["rows"])
    assert all(
        {"task_id", "module", "mechanism", "exact_path"}.issubset(row) for row in census["rows"]
    )

    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    (fixture_root / "research-roadmap.yaml").write_text(
        json.dumps(
            {
                "tasks": [
                    {"id": "not-a-task", "deliverable": "results/nope.json"},
                    {"id": "exp6197-later", "deliverable": "results/nope.json"},
                    {
                        "id": "exp6184-v536-evidence-isolation-preflight",
                        "title": "Preflight",
                        "track": "infrastructure",
                        "deliverable": (
                            "results/experiment_6184_v536_evidence_isolation_preflight.json"
                        ),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    assert [row["task_id"] for row in mod.declared_v536_tasks(fixture_root)] == [
        "exp6184-v536-evidence-isolation-preflight"
    ]


def test_scenario_report_6184_frozen_manifest_names_exact_invocation() -> None:
    """SCENARIO-REPORT-6184-FROZEN-PREFLIGHT-INVOCATION: invocation is reusable."""
    manifest = mod.build_frozen_preflight_manifest()

    assert manifest["version"] == mod.PREFLIGHT_CONTRACT_VERSION
    assert manifest["preflight_module"] == (
        "carnot.experiment_6184_v536_evidence_isolation_preflight"
    )
    assert manifest["pytest_target"] == (
        "tests/python/test_experiment_6184_v536_evidence_isolation_preflight.py"
    )
    assert manifest["artifact_root_env"] == "CARNOT_EXPERIMENT_ARTIFACT_ROOT"
    assert manifest["repository_wide_closure_claimed"] is False
    assert "mktemp -d /tmp/carnot-6184-preflight-" in manifest["canonical_task_owned_invocation"]
    assert "--collect-only" in manifest["commands"]["collection"]
    assert "coverage run --source=python/carnot" in manifest["commands"]["new_code_coverage_run"]
    assert (
        "--include='python/carnot/experiment_6184_v536_evidence_isolation_preflight.py'"
        in manifest["commands"]["new_code_coverage_report"]
    )


def test_scenario_report_6184_controls_classify_intercept_without_violation(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6184-INTERCEPTED-VS-MUTATION: blocked write is separate."""
    controls = mod.run_writer_controls(REPO, tmp_path / "preflight-root")

    assert controls["task_owned_temp_root"]["validated"] is True
    assert controls["canonical_writer"]["under_task_root"] is True
    assert controls["legacy_literal_writer"]["under_task_root"] is True
    assert controls["legacy_atomic_replace"]["under_task_root"] is True
    assert controls["checkpoint_resume"]["resumed_step"] == 8
    assert controls["subprocess"]["exit_code"] == 0
    assert controls["subprocess"]["canonical_exists"] is True
    assert controls["subprocess"]["legacy_exists"] is True
    assert controls["expected_intercepted_attempt"]["classification"] == (
        "expected_intercepted_attempt"
    )
    assert controls["expected_intercepted_attempt"]["counts_as_isolation_violation"] is False
    assert controls["actual_mutation"]["actual_mutation_count"] == 0
    assert controls["actual_mutation"]["tracked_result_mutated"] is False
    assert controls["traversal"]["raised"] is True
    assert controls["symlink_escape"]["raised"] is True
    assert controls["invalid_roots"]["workspace_root"]["raised"] is True
    assert controls["invalid_roots"]["repository_root"]["raised"] is True
    assert controls["invalid_roots"]["production_results_root"]["raised"] is True
    assert controls["invalid_roots"]["broad_tmp_root"]["raised"] is True
    assert controls["atomic_writer"]["leftover_tmp_files"] == []
    assert controls["quarantine_preservation"]["unchanged"] is True


def test_scenario_report_6184_artifact_schema_and_ready_score(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6184-SCHEMA-READINESS: artifact fields are auditable."""
    receipts = [
        {
            "name": "focused",
            "command": (
                ".venv/bin/pytest "
                "tests/python/test_experiment_6184_v536_evidence_isolation_preflight.py -q"
            ),
            "exit_code": 0,
            "stdout": "passed",
            "stderr": "",
        }
    ]
    artifact = mod.build_artifact(
        REPO,
        command_receipts=receipts,
        precondition_snapshot_path=_write_current_precondition_snapshot(tmp_path / "pre.json"),
        duration_s=0.5,
        temp_root=tmp_path / "artifact-root",
    )

    assert artifact["scope_boundary"]["qualified_scope"] == (
        "Exp6183-Exp6196 declared .536 writer/test surfaces only"
    )
    assert artifact["repository_wide_closure_claimed"] is False
    assert artifact["expected_intercepted_attempt_controls"]["attempt_count"] == 1
    assert artifact["expected_intercepted_attempt_controls"]["counts_as_violation_count"] == 0
    assert artifact["actual_mutation_controls"]["actual_mutation_count"] == 0
    assert artifact["v536_task_artifact_isolation_ready_score"] == 1
    assert artifact["isolation_violation_count"] == 0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_6184_failure_accounting_and_validation_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6184-SCHEMA-READINESS: failures force not-ready."""
    classified = mod.classify_command_receipts(
        [
            {"name": "ok", "exit_code": 0},
            {
                "name": "intercept",
                "exit_code": 1,
                "classification": "expected_intercepted_attempt",
            },
            {"name": "focused", "exit_code": 1, "stderr": "experiment_6184 failed"},
            {"name": "guard", "exit_code": 1, "stderr": "TrackedResultWriteError"},
            {"name": "unrelated", "exit_code": 2, "stderr": "legacy_fixture missing"},
            {"name": "regression", "exit_code": 1, "stderr": "new_regression"},
            {"name": "mystery", "exit_code": 1, "stderr": "unexpected"},
        ]
    )

    assert classified["counts"]["zero"] == 1
    assert classified["counts"]["expected_intercepted_attempt"] == 2
    assert classified["counts"]["task_scope_failure"] == 1
    assert classified["counts"]["unrelated_preexisting"] == 1
    assert classified["counts"]["new_regression"] == 1
    assert classified["counts"]["unclassified_task_scope_failure"] == 1
    assert (
        mod.count_isolation_violations(
            actual_mutation_count=0,
            escape_failure_count=0,
            classification=classified,
        )
        == 1
    )
    assert mod.actual_mutation_count({"caught_before_mutation": True}) == 0
    assert (
        mod.actual_mutation_count(
            {"target": "results/example.json", "caught_before_mutation": False}
        )
        == 1
    )
    assert mod.escape_failure_count({"traversal": {"raised": False}}) == 1

    manifest = mod.build_frozen_preflight_manifest()
    assert (
        mod.ready_score(
            actual_mutation_count=0,
            escape_failure_count=0,
            classification={"counts": {}},
            invocation_manifest=manifest,
            command_receipts=[{"exit_code": 0}],
        )
        == 1
    )
    assert (
        mod.ready_score(
            actual_mutation_count=1,
            escape_failure_count=0,
            classification={"counts": {}},
            invocation_manifest=manifest,
            command_receipts=[{"exit_code": 0}],
        )
        == 0
    )
    assert (
        mod.ready_score(
            actual_mutation_count=0,
            escape_failure_count=1,
            classification={"counts": {}},
            invocation_manifest=manifest,
            command_receipts=[{"exit_code": 0}],
        )
        == 0
    )
    assert (
        mod.ready_score(
            actual_mutation_count=0,
            escape_failure_count=0,
            classification=classified,
            invocation_manifest=manifest,
            command_receipts=[{"exit_code": 1}],
        )
        == 0
    )
    bad_manifest = dict(manifest)
    bad_manifest["version"] = "old"
    assert (
        mod.ready_score(
            actual_mutation_count=0,
            escape_failure_count=0,
            classification={"counts": {}},
            invocation_manifest=bad_manifest,
            command_receipts=[{"exit_code": 0}],
        )
        == 0
    )
    bad_manifest = dict(manifest)
    bad_manifest["repository_wide_closure_claimed"] = True
    assert (
        mod.ready_score(
            actual_mutation_count=0,
            escape_failure_count=0,
            classification={"counts": {}},
            invocation_manifest=bad_manifest,
            command_receipts=[{"exit_code": 0}],
        )
        == 0
    )
    bad_manifest = dict(manifest)
    bad_manifest["canonical_task_owned_invocation"] = ""
    assert (
        mod.ready_score(
            actual_mutation_count=0,
            escape_failure_count=0,
            classification={"counts": {}},
            invocation_manifest=bad_manifest,
            command_receipts=[{"exit_code": 0}],
        )
        == 0
    )
    assert (
        mod.ready_score(
            actual_mutation_count=0,
            escape_failure_count=0,
            classification={"counts": {}},
            invocation_manifest=manifest,
            command_receipts=[],
        )
        == 0
    )

    valid = {field: "placeholder" for field in mod.REQUIRED_ARTIFACT_FIELDS}
    valid["scope_boundary"] = {"qualified_scope": "Exp6183-Exp6196"}
    valid["repository_wide_closure_claimed"] = False
    valid["inference_substrate"] = mod.INFERENCE_SUBSTRATE
    valid["field_provenance"] = mod._field_provenance()
    valid["honest_verdict"] = "complete_ready: placeholder"
    valid["v536_task_artifact_isolation_ready_score"] = 1
    valid["isolation_violation_count"] = 0
    valid["reproducibility_checksum"] = mod.payload_checksum(valid)
    assert mod.validate_artifact(valid) == []

    bad = dict(valid)
    bad.pop("status")
    bad["repository_wide_closure_claimed"] = True
    bad["inference_substrate"] = "wrong"
    bad["field_provenance"] = []
    bad["honest_verdict"] = "maybe"
    bad["v536_task_artifact_isolation_ready_score"] = 2
    bad["isolation_violation_count"] = 1
    bad["reproducibility_checksum"] = "wrong"
    errors = mod.validate_artifact(bad)
    assert "missing:status" in errors
    assert "repository_wide_closure_claimed" in errors
    assert "inference_substrate" in errors
    assert "field_provenance:not_mapping" in errors
    assert "honest_verdict_prefix" in errors
    assert "ready_score" in errors
    assert "reproducibility_checksum" in errors

    bad_provenance = dict(valid)
    bad_provenance["field_provenance"] = dict(valid["field_provenance"])
    bad_provenance["field_provenance"]["status"] = {"principle": "wrong"}
    bad_provenance["reproducibility_checksum"] = mod.payload_checksum(bad_provenance)
    assert "field_provenance:status" in mod.validate_artifact(bad_provenance)

    bad_ready = dict(valid)
    bad_ready["isolation_violation_count"] = 1
    bad_ready["reproducibility_checksum"] = mod.payload_checksum(bad_ready)
    assert "ready_score_vs_violations" in mod.validate_artifact(bad_ready)

    assert mod.load_precondition_snapshot(tmp_path / "missing.json")["present"] is False
    list_snapshot = tmp_path / "list.json"
    list_snapshot.write_text("[]", encoding="utf-8")
    assert mod.load_precondition_snapshot(list_snapshot)["error"] == "not_json_object"
    assert mod._load_command_receipts(None) == []
    assert mod._raises_artifact_error(lambda: None)["raised"] is False
    receipts = tmp_path / "bad-receipts.json"
    receipts.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        mod._load_command_receipts(receipts)

    with pytest.MonkeyPatch.context() as patch:
        patch.delenv(mod.ARTIFACT_ROOT_ENV, raising=False)
        with mod._temporary_artifact_env(REPO, tmp_path / "env-root"):
            assert Path.cwd() == REPO
        assert mod.ARTIFACT_ROOT_ENV not in __import__("os").environ

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            mod.base_preflight,
            "SENTINEL_RESULT_PATHS",
            (Path("results/not-present-6184.json"),),
        )
        patch.setattr(
            mod.base_preflight,
            "_tracked_results",
            lambda _repo: [Path("results/fallback-6184.json")],
        )
        assert mod._choose_tracked_sentinel(REPO) == REPO / "results/fallback-6184.json"

    def _fake_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"ok": True}) + "\n",
            stderr="",
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PYTHONPATH", "fixture")
        patch.setattr(mod.subprocess, "run", _fake_run)
        sub = mod._run_subprocess_control(REPO, tmp_path / "subprocess-env-root")
    assert sub["parsed"] == {"ok": True}

    monkeypatch.setattr(mod, "validate_artifact", lambda _payload: ["forced"])
    with pytest.raises(ValueError):
        mod.build_artifact(
            REPO,
            command_receipts=[{"name": "focused", "command": "pytest", "exit_code": 0}],
            temp_root=tmp_path / "forced-error-root",
        )


def test_scenario_report_6184_main_writes_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6184-ESCAPE-ROOT-ATOMIC-QUARANTINE: CLI writes output."""
    receipts = tmp_path / "receipts.json"
    output = tmp_path / "experiment_6184_artifact.json"
    receipts.write_text(
        json.dumps(
            [
                {
                    "name": "focused",
                    "command": "pytest preflight",
                    "exit_code": 0,
                    "stdout": "ok",
                    "stderr": "",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = mod.main(
        [
            "--repo-root",
            str(REPO),
            "--command-receipts-json",
            str(receipts),
            "--precondition-snapshot",
            str(_write_current_precondition_snapshot(tmp_path / "pre-main.json")),
            "--output-path",
            str(output),
            "--duration-s",
            "0.25",
        ]
    )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert mod.validate_artifact(artifact) == []

    written: list[Path] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mod, "build_artifact", lambda *_args, **_kwargs: {"status": "ok"})
        patch.setattr(mod, "write_artifact", lambda _payload, path: written.append(path) or path)
        assert mod.main(["--repo-root", str(tmp_path), "--output-path", "relative.json"]) == 0
    assert written == [tmp_path / "relative.json"]
