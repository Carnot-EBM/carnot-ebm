"""Exp6170 task-scoped artifact isolation canary tests.

Spec refs:
  REQ-REPORT-6170
  SCENARIO-REPORT-6170-TASK-SCOPE-NON-CLOSURE
  SCENARIO-REPORT-6170-FROZEN-CANARY-INVOCATION
  SCENARIO-REPORT-6170-COMPATIBLE-WRITERS
  SCENARIO-REPORT-6170-ADVERSARIAL-CONTROLS
  SCENARIO-REPORT-6170-QUARANTINE-PRESERVATION
  SCENARIO-REPORT-6170-SCHEMA-READINESS
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from carnot import experiment_6170_v535_task_artifact_isolation_canary as mod


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
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_req_report_6170_spec_declares_task_scoped_canary_contract() -> None:
    """REQ-REPORT-6170: OpenSpec names the bounded canary contract."""
    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6170") :]
    section = section[: section.index("## Implementation Status (REQ-REPORT-6170)")]
    for required in [
        "Exp6169 through Exp6182",
        "repository-wide isolation",
        "canonical artifact resolver",
        "legacy relative",
        "task-owned temporary",
        "tracked result file must be caught before mutation",
        "symlink escape",
        "quarantine",
        "deterministic_task_scoped_repository_test_isolation",
        "SCENARIO-REPORT-6170-TASK-SCOPE-NON-CLOSURE",
        "SCENARIO-REPORT-6170-FROZEN-CANARY-INVOCATION",
        "SCENARIO-REPORT-6170-COMPATIBLE-WRITERS",
        "SCENARIO-REPORT-6170-ADVERSARIAL-CONTROLS",
        "SCENARIO-REPORT-6170-QUARANTINE-PRESERVATION",
        "SCENARIO-REPORT-6170-SCHEMA-READINESS",
    ]:
        assert required in section


def test_scenario_report_6170_task_census_is_bounded_to_declared_v535_tasks() -> None:
    """SCENARIO-REPORT-6170-TASK-SCOPE-NON-CLOSURE: census is not repo-wide."""
    census = mod.collect_v535_task_writer_census(REPO)

    assert census["scope"]["first_task"] == "exp6169-v535-transition"
    assert census["scope"]["last_task"] == "exp6182-v535-capstone-reconciliation"
    assert census["scope"]["repository_wide_writer_scan"] is False
    assert census["declared_task_count"] == 14
    assert {row["task_id"] for row in census["rows"]} == set(mod.V535_TASK_IDS)
    assert all(row["exact_path"].startswith("results/experiment_") for row in census["rows"])
    assert all("scripts/research_conductor.py" not in row["module"] for row in census["rows"])


def test_scenario_report_6170_frozen_manifest_names_exact_invocation() -> None:
    """SCENARIO-REPORT-6170-FROZEN-CANARY-INVOCATION: invocation is reusable."""
    manifest = mod.build_frozen_invocation_manifest()

    assert manifest["version"] == mod.CANARY_CONTRACT_VERSION
    assert manifest["canary_module"] == "carnot.experiment_6170_v535_task_artifact_isolation_canary"
    assert manifest["pytest_target"] == (
        "tests/python/test_experiment_6170_v535_task_artifact_isolation_canary.py"
    )
    assert manifest["artifact_root_env"] == "CARNOT_EXPERIMENT_ARTIFACT_ROOT"
    assert manifest["repository_wide_closure_claimed"] is False
    assert "--collect-only" in manifest["commands"]["collection"]
    assert "coverage run --source=python/carnot" in manifest["commands"]["new_code_coverage_run"]
    assert (
        "--include='python/carnot/experiment_6170_v535_task_artifact_isolation_canary.py'"
        in (manifest["commands"]["new_code_coverage_report"])
    )


def test_scenario_report_6170_writer_controls_redirect_and_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6170-COMPATIBLE-WRITERS: canary writers use a temp root."""
    controls = mod.run_writer_controls(REPO, tmp_path / "canary-root")

    assert controls["task_owned_temp_root"]["validated"] is True
    assert controls["canonical_writer"]["under_task_root"] is True
    assert controls["legacy_literal_writer"]["under_task_root"] is True
    assert controls["legacy_atomic_replace"]["under_task_root"] is True
    assert controls["checkpoint_resume"]["resumed_step"] == 7
    assert controls["subprocess"]["exit_code"] == 0
    assert controls["attempted_tracked_write"]["caught_before_mutation"] is True
    assert controls["traversal"]["raised"] is True
    assert controls["symlink_escape"]["raised"] is True
    assert controls["invalid_roots"]["repository_root"]["raised"] is True
    assert controls["invalid_roots"]["production_results_root"]["raised"] is True
    assert controls["invalid_roots"]["broad_tmp_root"]["raised"] is True
    assert controls["atomic_writer"]["leftover_tmp_files"] == []
    assert controls["quarantine_preservation"]["unchanged"] is True


def test_scenario_report_6170_artifact_schema_and_ready_score(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6170-SCHEMA-READINESS: artifact fields are auditable."""
    receipts = [
        {
            "name": "focused",
            "command": ".venv/bin/pytest tests/python/test_experiment_6170_v535_task_artifact_isolation_canary.py -q",
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

    assert (
        artifact["scope_boundary_and_repository_wide_closure_claimed"][
            "repository_wide_closure_claimed"
        ]
        is False
    )
    assert artifact["v535_task_artifact_isolation_ready_score"] == 1
    assert artifact["isolation_violation_count"] == 0
    assert artifact["inference_substrate"] == (
        "deterministic_task_scoped_repository_test_isolation"
    )
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_6170_failure_classification_forces_not_ready() -> None:
    """SCENARIO-REPORT-6170-SCHEMA-READINESS: nonzero canary failures zero readiness."""
    classified = mod.classify_command_receipts(
        [
            {
                "name": "canary-focused",
                "command": "pytest tests/python/test_experiment_6170_v535_task_artifact_isolation_canary.py",
                "exit_code": 1,
                "stderr": "AssertionError: canary failed",
            },
            {
                "name": "known-unrelated",
                "command": "pytest tests/python",
                "exit_code": 2,
                "stderr": "ModuleNotFoundError: No module named 'legacy_fixture'",
            },
            {
                "name": "mystery",
                "command": "pytest tests/python",
                "exit_code": 1,
                "stderr": "unexpected",
            },
        ]
    )

    assert classified["counts"]["canary_scope"] == 1
    assert classified["counts"]["unrelated_preexisting"] == 1
    assert classified["counts"]["unclassified"] == 1
    assert (
        mod.ready_score(
            tracked_mutation_count=0,
            classification=classified,
            invocation_manifest=mod.build_frozen_invocation_manifest(),
            command_receipts=[{"exit_code": 1}],
        )
        == 0
    )


def test_scenario_report_6170_helper_fallbacks_and_error_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6170-SCHEMA-READINESS: fallback branches are covered."""
    monkeypatch.setattr(mod, "_git", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()))
    assert mod._git_or_empty(tmp_path, ["status"]) == ""
    assert mod._status_paths(["??"]) == []
    assert mod._tracked_results(tmp_path) == []
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "fallback.json").write_text("{}", encoding="utf-8")
    assert mod._tracked_results(tmp_path) == [Path("results/fallback.json")]

    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    assert mod._read_json_object(scalar) == {"_non_object_json": "list"}
    assert mod.load_precondition_snapshot(tmp_path / "missing.json")["present"] is False
    list_snapshot = tmp_path / "list.json"
    list_snapshot.write_text("[]", encoding="utf-8")
    assert mod.load_precondition_snapshot(list_snapshot)["error"] == "not_json_object"
    assert mod._task_number("bad") is None
    assert mod._task_number("expabcd") is None

    roadmap = {
        "tasks": [
            {"id": "not-a-task", "deliverable": "results/nope.json"},
            {"id": "exp9999-later", "deliverable": "results/nope.json"},
            {
                "id": "exp6170-v535-task-artifact-isolation-canary",
                "title": "Canary",
                "track": "infrastructure",
                "deliverable": "results/experiment_6170_v535_task_artifact_isolation_canary.json",
            },
        ]
    }
    (tmp_path / "research-roadmap.yaml").write_text(json.dumps(roadmap), encoding="utf-8")
    assert [row["task_id"] for row in mod.declared_v535_tasks(tmp_path)] == [
        "exp6170-v535-task-artifact-isolation-canary"
    ]

    unknown = tmp_path / "unknown.py"
    unknown.write_text("x = 1\n", encoding="utf-8")
    assert (
        mod._writer_mechanism(tmp_path, Path("unknown.py"), "results/unknown.json")
        == "declared_module_without_detected_writer"
    )
    assert mod._raises_artifact_error(lambda: None)["raised"] is False
    assert mod._choose_tracked_sentinel(tmp_path) == tmp_path / "results/fallback.json"
    assert (
        mod._task_start_matrix_from_precondition(
            {},
            {"tracked_results": {}, "sentinel_hashes": {}, "quarantine_fields": {}},
        )["available"]
        is False
    )

    env_root = tmp_path / "env-root"
    env_root.mkdir()
    monkeypatch.delenv(mod.ARTIFACT_ROOT_ENV, raising=False)
    with mod._temporary_artifact_env(tmp_path, env_root):
        assert Path.cwd() == tmp_path
    assert mod.ARTIFACT_ROOT_ENV not in dict(__import__("os").environ)

    def _fake_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"ok": True}) + "\n",
            stderr="",
        )

    monkeypatch.setenv("PYTHONPATH", "fixture")
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    sub = mod._run_subprocess_control(REPO, tmp_path / "subproc-root")
    assert sub["parsed"] == {"ok": True}

    classified = mod.classify_command_receipts(
        [
            {"name": "explicit", "exit_code": 1, "classification": "new_regression"},
            {"name": "guard", "exit_code": 1, "stderr": "TrackedResultWriteError"},
            {"name": "new", "exit_code": 1, "stderr": "new_regression"},
        ]
    )
    assert classified["counts"]["new_regression"] == 2
    assert classified["counts"]["canary_scope"] == 1
    manifest = mod.build_frozen_invocation_manifest()
    assert (
        mod.ready_score(
            tracked_mutation_count=0,
            classification={"counts": {}},
            invocation_manifest=manifest,
            command_receipts=[],
        )
        == 0
    )
    assert (
        mod.ready_score(
            tracked_mutation_count=1,
            classification={"counts": {}},
            invocation_manifest=manifest,
            command_receipts=[{"exit_code": 0}],
        )
        == 0
    )
    bad_claim = dict(manifest)
    bad_claim["repository_wide_closure_claimed"] = True
    assert (
        mod.ready_score(
            tracked_mutation_count=0,
            classification={"counts": {}},
            invocation_manifest=bad_claim,
            command_receipts=[{"exit_code": 0}],
        )
        == 0
    )
    bad_version = dict(manifest)
    bad_version["version"] = "old"
    assert (
        mod.ready_score(
            tracked_mutation_count=0,
            classification={"counts": {}},
            invocation_manifest=bad_version,
            command_receipts=[{"exit_code": 0}],
        )
        == 0
    )


def test_scenario_report_6170_validation_and_cli_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6170-SCHEMA-READINESS: validators fail closed."""
    valid = {field: "placeholder" for field in mod.REQUIRED_ARTIFACT_FIELDS}
    valid["scope_boundary_and_repository_wide_closure_claimed"] = {
        "repository_wide_closure_claimed": False
    }
    valid["inference_substrate"] = mod.INFERENCE_SUBSTRATE
    valid["field_provenance"] = mod._field_provenance()
    valid["honest_verdict"] = "complete_ready: placeholder"
    valid["v535_task_artifact_isolation_ready_score"] = 1
    valid["isolation_violation_count"] = 0
    valid["reproducibility_checksum"] = mod.payload_checksum(valid)

    bad = dict(valid)
    bad.pop("status")
    bad["inference_substrate"] = "wrong"
    bad["scope_boundary_and_repository_wide_closure_claimed"] = {
        "repository_wide_closure_claimed": True
    }
    bad["field_provenance"] = []
    bad["honest_verdict"] = "maybe"
    bad["v535_task_artifact_isolation_ready_score"] = 3
    bad["isolation_violation_count"] = 1
    bad["reproducibility_checksum"] = "wrong"
    errors = mod.validate_artifact(bad)
    assert "missing:status" in errors
    assert "inference_substrate" in errors
    assert "repository_wide_closure_claimed" in errors
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

    assert mod._load_command_receipts(None) == []
    receipts = tmp_path / "bad-receipts.json"
    receipts.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        mod._load_command_receipts(receipts)

    def _fake_build(*_args: object, **_kwargs: object) -> dict[str, object]:
        return valid

    written: list[Path] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mod, "build_artifact", _fake_build)
        patch.setattr(mod, "write_artifact", lambda _payload, path: written.append(path) or path)
        assert mod.main(["--repo-root", str(tmp_path), "--output-path", "relative.json"]) == 0
    assert written == [tmp_path / "relative.json"]

    monkeypatch.setattr(mod, "validate_artifact", lambda _payload: ["forced"])
    with pytest.raises(ValueError):
        mod.build_artifact(
            REPO,
            command_receipts=[{"name": "focused", "command": "pytest", "exit_code": 0}],
            temp_root=tmp_path / "forced-error-root",
        )


def test_scenario_report_6170_main_writes_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6170-QUARANTINE-PRESERVATION: CLI writes only requested output."""
    receipts = tmp_path / "receipts.json"
    output = tmp_path / "experiment_6170_artifact.json"
    receipts.write_text(
        json.dumps(
            [
                {
                    "name": "focused",
                    "command": "pytest canary",
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
